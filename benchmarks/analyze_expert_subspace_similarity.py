#!/usr/bin/env python3
"""Measure same-layer expert subspace overlap for safetensors MoE checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def parse_csv_strs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(q: float) -> float:
        idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return ordered[idx]

    return {
        "min": ordered[0],
        "p10": pick(0.10),
        "p25": pick(0.25),
        "median": pick(0.50),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "max": ordered[-1],
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def load_config(model_path: Path) -> dict[str, Any]:
    with (model_path / "config.json").open() as handle:
        return json.load(handle)


def load_weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    with index_path.open() as handle:
        index = json.load(handle)
    return index["weight_map"]


def read_tensor(model_path: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard = model_path / weight_map[key]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def expert_weight_key(layer: int, expert: int, matrix: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{matrix}.weight"


def ambient_dim(config: dict[str, Any], matrix: str, space: str) -> int:
    hidden_size = int(config["hidden_size"])
    moe_intermediate_size = int(config["moe_intermediate_size"])
    if matrix in {"gate_proj", "up_proj"}:
        return hidden_size if space == "input" else moe_intermediate_size
    if matrix == "down_proj":
        return moe_intermediate_size if space == "input" else hidden_size
    raise ValueError(f"Unsupported matrix: {matrix}")


def compute_bases(
    weight: torch.Tensor,
    max_rank: int,
    device: torch.device,
) -> dict[str, torch.Tensor | float | list[float]]:
    w = weight.to(device=device, dtype=torch.float32)
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    rank = min(max_rank, s.numel())
    energy = s.square()
    total_energy = float(energy.sum().item())
    result: dict[str, torch.Tensor | float | list[float]] = {
        "input": vh[:rank, :].T.contiguous().cpu(),
        "output": u[:, :rank].contiguous().cpu(),
        "singular_values": s[:rank].detach().cpu().tolist(),
        "total_energy": total_energy,
    }
    del w, u, s, vh
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def sampled_pairs(n: int, max_pairs: int, seed: int) -> list[tuple[int, int]]:
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if max_pairs <= 0 or max_pairs >= len(all_pairs):
        return all_pairs
    rng = random.Random(seed)
    return rng.sample(all_pairs, max_pairs)


def summarize_pairs(
    bases: list[torch.Tensor],
    rank: int,
    pairs: list[tuple[int, int]],
    device: torch.device,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    stacked = torch.stack([basis[:, :rank] for basis in bases], dim=0).to(device=device)
    pair_rows: list[dict[str, float | int]] = []
    spectra_rows: list[dict[str, float | int]] = []
    spectra_accum = [[] for _ in range(rank)]

    for expert_i, expert_j in pairs:
        m = stacked[expert_i].T @ stacked[expert_j]
        sigmas = torch.linalg.svdvals(m).clamp_(0.0, 1.0)
        angles = torch.acos(sigmas)
        cos2_sum = float(sigmas.square().sum().item())
        projection_fro = math.sqrt(max(0.0, 2.0 * (rank - cos2_sum)))
        grassmann = math.sqrt(float(angles.square().sum().item()))
        mean_angle_deg = float(angles.mean().item() * 180.0 / math.pi)
        pair_rows.append(
            {
                "expert_i": expert_i,
                "expert_j": expert_j,
                "rank": rank,
                "projection_overlap": cos2_sum / rank,
                "projection_fro": projection_fro,
                "grassmann": grassmann,
                "mean_principal_angle_deg": mean_angle_deg,
                "max_principal_angle_deg": float(angles.max().item() * 180.0 / math.pi),
                "min_principal_angle_deg": float(angles.min().item() * 180.0 / math.pi),
            }
        )
        for idx, sigma in enumerate(sigmas.detach().cpu().tolist()):
            spectra_accum[idx].append(float(sigma))

    for idx, values in enumerate(spectra_accum):
        stats = quantiles(values)
        spectra_rows.append(
            {
                "rank": rank,
                "principal_index": idx + 1,
                "cos_mean": stats["mean"],
                "cos_p10": stats["p10"],
                "cos_median": stats["median"],
                "cos_p90": stats["p90"],
            }
        )

    del stacked
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pair_rows, spectra_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_heatmap(
    output_dir: Path,
    rows: list[dict[str, Any]],
    layer: int,
    matrix: str,
    space: str,
    rank: int,
    num_experts: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    heatmap = np.full((num_experts, num_experts), np.nan, dtype=float)
    for row in rows:
        i = int(row["expert_i"])
        j = int(row["expert_j"])
        value = float(row["projection_overlap"])
        heatmap[i, j] = value
        heatmap[j, i] = value
    np.fill_diagonal(heatmap, 1.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(heatmap, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(f"layer {layer} {matrix} {space} rank {rank}")
    ax.set_xlabel("expert")
    ax.set_ylabel("expert")
    fig.colorbar(im, ax=ax, label="projection overlap")
    fig.tight_layout()
    fig.savefig(output_dir / f"heatmap_layer{layer:02d}_{matrix}_{space}_rank{rank}.png", dpi=160)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(model_path)
    weight_map = load_weight_map(model_path)
    layers = parse_int_selection(args.layers)
    ranks = sorted(parse_csv_ints(args.ranks))
    matrices = parse_csv_strs(args.matrices)
    spaces = parse_csv_strs(args.spaces)
    max_rank = max(ranks)
    num_experts = int(args.expert_limit or cfg["n_routed_experts"])
    device = torch.device(args.device)

    pair_rows_all: list[dict[str, Any]] = []
    spectra_rows_all: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "model_path": str(model_path),
        "config": {
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "first_k_dense_replace": cfg.get("first_k_dense_replace"),
            "n_routed_experts": cfg.get("n_routed_experts"),
            "num_experts_per_tok": cfg.get("num_experts_per_tok"),
            "hidden_size": cfg.get("hidden_size"),
            "moe_intermediate_size": cfg.get("moe_intermediate_size"),
            "torch_dtype": cfg.get("torch_dtype"),
        },
        "layers": layers,
        "ranks": ranks,
        "matrices": matrices,
        "spaces": spaces,
        "num_experts_analyzed": num_experts,
        "device": str(device),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [],
    }

    for layer in layers:
        for matrix in matrices:
            print(f"[load] layer={layer} matrix={matrix}", flush=True)
            expert_bases: dict[str, list[torch.Tensor]] = {space: [] for space in spaces}
            singular_values: list[list[float]] = []
            total_energies: list[float] = []

            for expert in range(num_experts):
                key = expert_weight_key(layer, expert, matrix)
                if key not in weight_map:
                    raise KeyError(f"Missing tensor: {key}")
                bases = compute_bases(read_tensor(model_path, weight_map, key), max_rank, device)
                for space in spaces:
                    expert_bases[space].append(bases[space])  # type: ignore[arg-type]
                singular_values.append(bases["singular_values"])  # type: ignore[arg-type]
                total_energies.append(float(bases["total_energy"]))

            for rank in ranks:
                for expert, values in enumerate(singular_values):
                    top_energy = sum(v * v for v in values[:rank])
                    total_energy = total_energies[expert]
                    energy_rows.append(
                        {
                            "layer": layer,
                            "matrix": matrix,
                            "expert": expert,
                            "rank": rank,
                            "top_energy_fraction": top_energy / total_energy if total_energy else 0.0,
                        }
                    )

            pairs = sampled_pairs(num_experts, args.max_pairs, args.seed)
            for space in spaces:
                for rank in ranks:
                    rank = min(rank, expert_bases[space][0].shape[1])
                    print(
                        f"[pairs] layer={layer} matrix={matrix} space={space} rank={rank} pairs={len(pairs)}",
                        flush=True,
                    )
                    pair_rows, spectra_rows = summarize_pairs(expert_bases[space], rank, pairs, device)
                    for row in pair_rows:
                        dim = ambient_dim(cfg, matrix, space)
                        random_overlap = rank / dim
                        observed_overlap = float(row["projection_overlap"])
                        row.update(
                            {
                                "layer": layer,
                                "matrix": matrix,
                                "space": space,
                                "ambient_dim": dim,
                                "random_projection_overlap": random_overlap,
                                "projection_overlap_enrichment": (
                                    observed_overlap / random_overlap if random_overlap else 0.0
                                ),
                            }
                        )
                    for row in spectra_rows:
                        row.update({"layer": layer, "matrix": matrix, "space": space})
                    pair_rows_all.extend(pair_rows)
                    spectra_rows_all.extend(spectra_rows)

                    metrics = {
                        "layer": layer,
                        "matrix": matrix,
                        "space": space,
                        "rank": rank,
                        "ambient_dim": ambient_dim(cfg, matrix, space),
                        "random_projection_overlap": rank / ambient_dim(cfg, matrix, space),
                        "num_pairs": len(pair_rows),
                        "projection_overlap": quantiles([float(r["projection_overlap"]) for r in pair_rows]),
                        "projection_overlap_enrichment": quantiles(
                            [float(r["projection_overlap_enrichment"]) for r in pair_rows]
                        ),
                        "projection_fro": quantiles([float(r["projection_fro"]) for r in pair_rows]),
                        "grassmann": quantiles([float(r["grassmann"]) for r in pair_rows]),
                        "mean_principal_angle_deg": quantiles(
                            [float(r["mean_principal_angle_deg"]) for r in pair_rows]
                        ),
                    }
                    summary["results"].append(metrics)

                    if args.write_heatmaps and (not args.heatmap_ranks or rank in parse_csv_ints(args.heatmap_ranks)):
                        maybe_write_heatmap(output_root, pair_rows, layer, matrix, space, rank, num_experts)

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (output_root / "subspace_similarity_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(output_root / "pairwise_subspace_metrics.csv", pair_rows_all)
    write_csv(output_root / "principal_angle_spectra.csv", spectra_rows_all)
    write_csv(output_root / "svd_energy.csv", energy_rows)

    lines = [
        "# Expert Subspace Similarity Summary",
        "",
        f"- Model: `{model_path}`",
        f"- Layers: `{layers}`",
        f"- Matrices: `{matrices}`",
        f"- Spaces: `{spaces}`",
        f"- Ranks: `{ranks}`",
        f"- Experts analyzed per layer: `{num_experts}`",
        "",
        "| Layer | Matrix | Space | Rank | Overlap mean | Random | Enrich mean | Grassmann mean | Mean angle deg |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["results"]:
        lines.append(
            "| {layer} | {matrix} | {space} | {rank} | {om:.4f} | {rand:.4f} | {enrich:.2f} | {gm:.4f} | {am:.2f} |".format(
                layer=item["layer"],
                matrix=item["matrix"],
                space=item["space"],
                rank=item["rank"],
                om=item["projection_overlap"]["mean"],
                rand=item["random_projection_overlap"],
                enrich=item["projection_overlap_enrichment"]["mean"],
                gm=item["grassmann"]["mean"],
                am=item["mean_principal_angle_deg"]["mean"],
            )
        )
    (output_root / "subspace_similarity_summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--layers", default="1,13,26", help="Comma-separated layer ids or ranges, e.g. 1,13,26 or 1-26")
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--matrices", default="gate_proj,up_proj,down_proj")
    parser.add_argument("--spaces", default="input,output", help="input, output, or both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expert-limit", type=int, default=0, help="debug only; 0 means all routed experts")
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means all expert pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-heatmaps", action="store_true")
    parser.add_argument("--heatmap-ranks", default="64", help="comma-separated ranks to plot; empty means all")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
