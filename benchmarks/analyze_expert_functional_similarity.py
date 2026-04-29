#!/usr/bin/env python3
"""Compare same-layer MoE expert outputs and correlate them with subspace overlap."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_infinity.utils.qwen_trace_builder import (  # noqa: E402
    count_chat_tokens,
    load_finemoe_lmsys_prompts,
    make_chat_messages,
)


def parse_int_selection(value: str) -> list[int]:
    selected: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            selected.append(int(item))
            continue
        start, end = item.split("-", 1)
        start_i = int(start.strip())
        end_i = int(end.strip())
        step = 1 if end_i >= start_i else -1
        selected.extend(range(start_i, end_i + step, step))
    return selected


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(q: float) -> float:
        idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return ordered[idx]

    return {
        "min": ordered[0],
        "p25": pick(0.25),
        "median": pick(0.50),
        "p75": pick(0.75),
        "max": ordered[-1],
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(x) != len(y):
        return 0.0
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denom_x = math.sqrt(sum(value * value for value in dx))
    denom_y = math.sqrt(sum(value * value for value in dy))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / (denom_x * denom_y)


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rank_values(x), rank_values(y))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class LayerSampleStore:
    def __init__(self, layers: Iterable[int], max_tokens_per_layer: int, seed: int) -> None:
        self.layers = set(layers)
        self.max_tokens_per_layer = max_tokens_per_layer
        self.seed = seed
        self.chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.layers}
        self.total_seen: dict[int, int] = {layer: 0 for layer in self.layers}

    def add(self, layer: int, hidden_states: torch.Tensor) -> None:
        if layer not in self.layers:
            return
        flat = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
        self.total_seen[layer] += int(flat.shape[0])
        chunk = flat.to(device="cpu", dtype=torch.float16)
        self.chunks[layer].append(chunk)
        self._trim(layer, oversample=4)

    def _trim(self, layer: int, oversample: int = 1) -> None:
        limit = self.max_tokens_per_layer * oversample
        total = sum(chunk.shape[0] for chunk in self.chunks[layer])
        if total <= limit:
            return
        merged = torch.cat(self.chunks[layer], dim=0)
        generator = torch.Generator(device="cpu").manual_seed(self.seed + layer + total)
        keep = torch.randperm(merged.shape[0], generator=generator)[: self.max_tokens_per_layer]
        self.chunks[layer] = [merged[keep].contiguous()]

    def finalize(self) -> dict[int, torch.Tensor]:
        samples: dict[int, torch.Tensor] = {}
        for layer in sorted(self.layers):
            if not self.chunks[layer]:
                samples[layer] = torch.empty(0)
                continue
            merged = torch.cat(self.chunks[layer], dim=0)
            if merged.shape[0] > self.max_tokens_per_layer:
                generator = torch.Generator(device="cpu").manual_seed(self.seed + layer)
                keep = torch.randperm(merged.shape[0], generator=generator)[: self.max_tokens_per_layer]
                merged = merged[keep]
            samples[layer] = merged.contiguous()
        return samples


def load_prompts(args: argparse.Namespace, tokenizer: Any) -> list[Any]:
    prompts = load_finemoe_lmsys_prompts(args.source_json)
    filtered = [
        item
        for item in prompts
        if count_chat_tokens(tokenizer, item.prompt) <= args.max_input_length
    ]
    rng = random.Random(args.seed)
    rng.shuffle(filtered)
    if len(filtered) < args.prompt_count:
        raise ValueError(f"Only {len(filtered)} prompts fit max_input_length={args.max_input_length}")
    return filtered[: args.prompt_count]


def render_prompt(tokenizer: Any, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            make_chat_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def collect_layer_samples(
    model: Any,
    tokenizer: Any,
    prompts: list[Any],
    layers: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    store = LayerSampleStore(layers, args.max_tokens_per_layer, args.seed)
    handles = []
    model_layers = model.model.layers

    for layer in layers:
        def hook(_module: Any, inputs: tuple[Any, ...], *, layer_idx: int = layer) -> None:
            if inputs:
                store.add(layer_idx, inputs[0])

        handles.append(model_layers[layer].mlp.register_forward_pre_hook(hook))

    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), args.batch_size):
                batch = prompts[start : start + args.batch_size]
                texts = [render_prompt(tokenizer, item.prompt) for item in batch]
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_length,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    return store.finalize(), dict(store.total_seen)


def compute_functional_rows(
    model: Any,
    layer: int,
    samples: torch.Tensor,
    expert_limit: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if samples.numel() == 0:
        return []
    experts = model.model.layers[layer].mlp.experts
    num_experts = expert_limit or len(experts)
    hidden = samples.to(device=device, dtype=torch.float16)
    outputs = []
    with torch.inference_mode():
        for expert_idx in range(num_experts):
            output = experts[expert_idx](hidden).detach().to(dtype=torch.float32)
            outputs.append(output.cpu())

    y = torch.stack(outputs, dim=0).to(device=device)
    y_norm = F.normalize(y, dim=-1)
    token_cos = torch.einsum("eth,fth->eft", y_norm, y_norm).mean(dim=-1)
    flat = y.reshape(y.shape[0], -1)
    norms = flat.square().sum(dim=1)
    dot = flat @ flat.T
    dist2 = (norms[:, None] + norms[None, :] - 2.0 * dot).clamp_min_(0.0)
    l2 = torch.sqrt(dist2)
    relative_denom = torch.sqrt(0.5 * (norms[:, None] + norms[None, :])).clamp_min_(1e-12)
    relative_l2 = l2 / relative_denom

    rows: list[dict[str, Any]] = []
    max_l2 = float(l2.max().item()) or 1.0
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            rows.append(
                {
                    "layer": layer,
                    "expert_i": i,
                    "expert_j": j,
                    "num_tokens": int(samples.shape[0]),
                    "mean_output_cosine": float(token_cos[i, j].item()),
                    "frobenius_l2": float(l2[i, j].item()),
                    "relative_l2": float(relative_l2[i, j].item()),
                    "frobenius_similarity": 1.0 - float(l2[i, j].item()) / max_l2,
                }
            )
    del y, y_norm, flat, norms, dot, dist2, l2, relative_l2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def load_subspace_metrics(path: str) -> dict[tuple[int, int, int], dict[str, float]]:
    if not path:
        return {}
    metrics: dict[tuple[int, int, int], dict[str, float]] = defaultdict(dict)
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            key = (int(row["layer"]), int(row["expert_i"]), int(row["expert_j"]))
            label = f"{row['matrix']}_{row['space']}_rank{row['rank']}"
            metrics[key][label] = float(row["projection_overlap"])
    return metrics


def compute_correlations(
    functional_rows: list[dict[str, Any]],
    subspace_metrics: dict[tuple[int, int, int], dict[str, float]],
    ranks: list[int],
) -> list[dict[str, Any]]:
    if not subspace_metrics:
        return []
    functional_by_key = {
        (int(row["layer"]), int(row["expert_i"]), int(row["expert_j"])): row
        for row in functional_rows
    }
    metric_names = sorted({name for values in subspace_metrics.values() for name in values})
    for rank in ranks:
        parts = [f"gate_proj_input_rank{rank}", f"up_proj_input_rank{rank}", f"down_proj_output_rank{rank}"]
        metric_names.append(f"composite_shared_rank{rank}")

    rows: list[dict[str, Any]] = []
    for metric_name in metric_names:
        x: list[float] = []
        y_cos: list[float] = []
        y_l2_sim: list[float] = []
        for key, functional in functional_by_key.items():
            values = subspace_metrics.get(key)
            if not values:
                continue
            if metric_name.startswith("composite_shared_rank"):
                rank = int(metric_name.rsplit("rank", 1)[1])
                parts = [
                    f"gate_proj_input_rank{rank}",
                    f"up_proj_input_rank{rank}",
                    f"down_proj_output_rank{rank}",
                ]
                if not all(part in values for part in parts):
                    continue
                score = statistics.fmean(values[part] for part in parts)
            else:
                if metric_name not in values:
                    continue
                score = values[metric_name]
            x.append(score)
            y_cos.append(float(functional["mean_output_cosine"]))
            y_l2_sim.append(-float(functional["relative_l2"]))
        if len(x) < 3:
            continue
        rows.append(
            {
                "subspace_metric": metric_name,
                "num_pairs": len(x),
                "pearson_output_cosine": pearson(x, y_cos),
                "spearman_output_cosine": spearman(x, y_cos),
                "pearson_neg_relative_l2": pearson(x, y_l2_sim),
                "spearman_neg_relative_l2": spearman(x, y_l2_sim),
            }
        )
    return rows


def analyze(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    layers = parse_int_selection(args.layers)
    ranks = parse_csv_ints(args.subspace_ranks)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = load_prompts(args, tokenizer)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    samples, total_seen = collect_layer_samples(model, tokenizer, prompts, layers, args, device)
    functional_rows: list[dict[str, Any]] = []
    for layer in layers:
        print(
            f"[functional] layer={layer} samples={tuple(samples[layer].shape)} seen={total_seen.get(layer, 0)}",
            flush=True,
        )
        functional_rows.extend(
            compute_functional_rows(model, layer, samples[layer], args.expert_limit, device)
        )
    write_csv(output_root / "expert_pair_functional_metrics.csv", functional_rows)

    subspace = load_subspace_metrics(args.subspace_pairwise_csv)
    correlation_rows = compute_correlations(functional_rows, subspace, ranks)
    write_csv(output_root / "subspace_function_correlation.csv", correlation_rows)

    by_layer_cos: dict[int, list[float]] = defaultdict(list)
    by_layer_l2: dict[int, list[float]] = defaultdict(list)
    for row in functional_rows:
        by_layer_cos[int(row["layer"])].append(float(row["mean_output_cosine"]))
        by_layer_l2[int(row["layer"])].append(float(row["relative_l2"]))

    summary = {
        "model_path": args.model_path,
        "source_json": args.source_json,
        "prompt_count": len(prompts),
        "max_input_length": args.max_input_length,
        "layers": layers,
        "expert_limit": args.expert_limit,
        "max_tokens_per_layer": args.max_tokens_per_layer,
        "total_seen_per_layer": total_seen,
        "functional_summary": {
            str(layer): {
                "mean_output_cosine": quantiles(by_layer_cos[layer]),
                "relative_l2": quantiles(by_layer_l2[layer]),
            }
            for layer in layers
        },
        "top_correlations_by_spearman_cosine": sorted(
            correlation_rows,
            key=lambda row: abs(float(row["spearman_output_cosine"])),
            reverse=True,
        )[:20],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with (output_root / "functional_similarity_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        "# Expert Functional Similarity Summary",
        "",
        f"- Model: `{args.model_path}`",
        f"- Prompts: `{len(prompts)}` from `{args.source_json}`",
        f"- Layers: `{layers}`",
        f"- Max tokens per layer: `{args.max_tokens_per_layer}`",
        "",
        "| Layer | Cosine mean | Cosine median | Relative L2 mean | Relative L2 median |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for layer in layers:
        cos_stats = summary["functional_summary"][str(layer)]["mean_output_cosine"]
        l2_stats = summary["functional_summary"][str(layer)]["relative_l2"]
        lines.append(
            f"| {layer} | {cos_stats.get('mean', 0.0):.4f} | {cos_stats.get('median', 0.0):.4f} | "
            f"{l2_stats.get('mean', 0.0):.4f} | {l2_stats.get('median', 0.0):.4f} |"
        )
    if correlation_rows:
        lines.extend(
            [
                "",
                "## Strongest Subspace/Function Correlations",
                "",
                "| Subspace metric | Pairs | Spearman cosine | Spearman -relL2 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in summary["top_correlations_by_spearman_cosine"][:12]:
            lines.append(
                f"| {row['subspace_metric']} | {row['num_pairs']} | "
                f"{row['spearman_output_cosine']:.4f} | {row['spearman_neg_relative_l2']:.4f} |"
            )
    (output_root / "functional_similarity_summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--layers", default="1,13,26", help="Comma-separated layer ids or ranges, e.g. 1,13,26")
    parser.add_argument("--prompt-count", type=int, default=256)
    parser.add_argument("--max-input-length", type=int, default=192)
    parser.add_argument("--max-tokens-per-layer", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--expert-limit", type=int, default=0, help="debug only; 0 means all routed experts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subspace-pairwise-csv", default="")
    parser.add_argument("--subspace-ranks", default="16,64,128")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
