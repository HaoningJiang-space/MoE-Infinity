from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

from moe_infinity import MoE
from moe_infinity.utils.qwen_benchmark import (
    QWEN_BENCHMARK_VARIANTS,
    aggregate_request_records,
    build_env_note,
    build_qwen_benchmark_config,
    diff_counter_dict,
    discover_trace_files,
    load_chat_trace,
    prepare_case_paths,
    render_markdown_summary,
    summarize_hit_rate_tensor,
)
from moe_infinity.utils.qwen_smoke import dispatcher_stats_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct-generate Qwen offloading benchmark for moe_infinity_fgo."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(QWEN_BENCHMARK_VARIANTS),
        choices=list(QWEN_BENCHMARK_VARIANTS),
    )
    parser.add_argument("--traces", nargs="+", default=None)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--measured-requests", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device-memory-ratio", type=float, default=0.6)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--library-capacity", type=int, default=32)
    parser.add_argument("--library-metric", default="cosine")
    parser.add_argument("--library-admission", default="diversity_aware")
    parser.add_argument("--backbone-topk", type=int, default=8)
    parser.add_argument("--prefetch-future-layers", type=int, default=4)
    parser.add_argument("--prefetch-max-candidates", type=int, default=32)
    parser.add_argument("--summary-json")
    parser.add_argument("--summary-md")
    parser.add_argument(
        "--require-single-visible-gpu",
        action="store_true",
        default=True,
    )
    return parser.parse_args()


def _default_trace_dir() -> Path:
    return Path(__file__).resolve().parent / "traces" / "qwen"


def _snapshot_library_stats(model: MoE) -> Dict[str, int]:
    policy = getattr(model.engine, "offloading_policy", None)
    if policy is None:
        return {}
    return dict(policy.library_stats())


def _snapshot_cache_hit_rate(model: MoE) -> Dict[str, float | int]:
    archer_engine = getattr(model.engine, "archer_engine", None)
    if archer_engine is None or not hasattr(archer_engine, "get_hit_rate"):
        return {}
    try:
        return summarize_hit_rate_tensor(archer_engine.get_hit_rate())
    except Exception:
        return {}


def _run_case(
    *,
    model_path: str,
    output_root: str,
    trace_name: str,
    trace_path: Path,
    variant: str,
    warmup_requests: int,
    measured_requests: int,
    max_new_tokens: int,
    device_memory_ratio: float,
    num_threads: int,
    library_capacity: int,
    library_metric: str,
    library_admission: str,
    backbone_topk: int,
    prefetch_future_layers: int,
    prefetch_max_candidates: int,
) -> Dict[str, Any]:
    case_paths = prepare_case_paths(
        output_root=output_root,
        trace_name=trace_name,
        variant=variant,
    )
    requests = load_chat_trace(trace_path)
    total_requests = warmup_requests + measured_requests
    if len(requests) < total_requests:
        raise ValueError(
            f"Trace '{trace_name}' only has {len(requests)} requests, "
            f"need at least {total_requests}"
        )
    selected_requests = requests[:total_requests]
    config = build_qwen_benchmark_config(
        variant=variant,
        offload_path=case_paths["offload_path"],
        device_memory_ratio=device_memory_ratio,
        num_threads=num_threads,
        library_capacity=library_capacity,
        library_metric=library_metric,
        library_admission=library_admission,
        backbone_topk=backbone_topk,
        prefetch_future_layers=prefetch_future_layers,
        prefetch_max_candidates=prefetch_max_candidates,
    )

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    model = MoE(model_path, config)
    dispatcher = model.engine.expert_dispatcher

    records: List[Dict[str, Any]] = []
    previous_library_stats: Dict[str, int] = {}
    previous_cache_stats: Dict[str, float | int] = {}

    for index, request in enumerate(selected_requests):
        if hasattr(dispatcher, "reset_runtime_stats"):
            dispatcher.reset_runtime_stats()
        prompt = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency_s = time.perf_counter() - start

        generated_ids = outputs[0][input_ids.shape[1] :]
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        dispatcher_stats = {}
        if hasattr(dispatcher, "get_runtime_stats"):
            dispatcher_stats = dispatcher_stats_dict(
                dispatcher.get_runtime_stats()
            )
        library_stats = _snapshot_library_stats(model)
        library_stats_delta = diff_counter_dict(
            library_stats,
            previous_library_stats,
            keys=(
                "query_count",
                "hit_count",
                "admit_count",
                "duplicate_update_count",
            ),
        )
        cache_hit_rate_snapshot = _snapshot_cache_hit_rate(model)
        cache_hit_rate_delta = diff_counter_dict(
            cache_hit_rate_snapshot,
            previous_cache_stats,
            keys=(
                "visit_count",
                "gpu_visit_count",
                "cpu_visit_count",
                "hit_count",
                "gpu_hit_count",
                "cpu_hit_count",
                "prefetch_count",
            ),
        )
        if cache_hit_rate_delta.get("visit_count", 0) > 0:
            cache_hit_rate_delta["overall_hit_rate"] = (
                cache_hit_rate_delta["hit_count"]
                / cache_hit_rate_delta["visit_count"]
            )
        else:
            cache_hit_rate_delta["overall_hit_rate"] = 0.0

        is_warmup = index < warmup_requests
        record = {
            "trace_name": trace_name,
            "variant": variant,
            "request_index": index,
            "request_id": request.request_id,
            "tag": request.tag,
            "is_warmup": is_warmup,
            "latency_s": latency_s,
            "generated_tokens": int(generated_ids.numel()),
            "latency_per_generated_token_ms": (
                (latency_s * 1000.0) / max(int(generated_ids.numel()), 1)
            ),
            "response_text_prefix": generated_text[:160],
            "dispatcher_stats": dispatcher_stats,
            "library_stats": library_stats,
            "library_stats_delta": library_stats_delta,
            "cache_hit_rate_snapshot": cache_hit_rate_snapshot,
            "cache_hit_rate_delta": cache_hit_rate_delta,
        }
        records.append(record)
        previous_library_stats = library_stats
        previous_cache_stats = cache_hit_rate_snapshot

    measured = [record for record in records if not record["is_warmup"]]
    aggregate = aggregate_request_records(measured)
    case_result = {
        "trace_name": trace_name,
        "trace_path": str(trace_path),
        "variant": variant,
        "config": config,
        "offload_path": case_paths["offload_path"],
        "warmup_requests": warmup_requests,
        "measured_requests": measured_requests,
        "max_new_tokens": max_new_tokens,
        "records": records,
        "aggregate": aggregate,
    }
    raw_path = Path(case_paths["raw_json"])
    raw_path.write_text(json.dumps(case_result, indent=2), encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return case_result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen offloading benchmark.")
    visible_count = torch.cuda.device_count()
    if args.require_single_visible_gpu and visible_count != 1:
        raise RuntimeError(
            f"Expected exactly 1 visible CUDA device, found {visible_count}. "
            "Use CUDA_VISIBLE_DEVICES to isolate a single GPU."
        )

    trace_dir = Path(args.trace_dir) if args.trace_dir else _default_trace_dir()
    discovered = discover_trace_files(trace_dir)
    trace_names = args.traces if args.traces else list(discovered)
    for trace_name in trace_names:
        if trace_name not in discovered:
            raise ValueError(
                f"Trace '{trace_name}' not found under {trace_dir}. "
                f"Available: {sorted(discovered)}"
            )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    env_note = build_env_note(
        model_path=args.model_path,
        offload_root=str(output_root / "offload"),
        visible_count=visible_count,
        repo_root=Path(__file__).resolve().parents[1],
    )
    (output_root / "env_note.json").write_text(
        json.dumps(env_note, indent=2),
        encoding="utf-8",
    )

    trace_results: Dict[str, Dict[str, Any]] = {}
    for trace_name in trace_names:
        trace_path = discovered[trace_name]
        trace_results[trace_name] = {}
        for variant in args.variants:
            case_result = _run_case(
                model_path=args.model_path,
                output_root=str(output_root),
                trace_name=trace_name,
                trace_path=trace_path,
                variant=variant,
                warmup_requests=args.warmup_requests,
                measured_requests=args.measured_requests,
                max_new_tokens=args.max_new_tokens,
                device_memory_ratio=args.device_memory_ratio,
                num_threads=args.num_threads,
                library_capacity=args.library_capacity,
                library_metric=args.library_metric,
                library_admission=args.library_admission,
                backbone_topk=args.backbone_topk,
                prefetch_future_layers=args.prefetch_future_layers,
                prefetch_max_candidates=args.prefetch_max_candidates,
            )
            trace_results[trace_name][variant] = case_result

    summary = {
        "env": env_note,
        "variants": list(args.variants),
        "traces": trace_names,
        "trace_results": trace_results,
    }
    summary_json = (
        Path(args.summary_json)
        if args.summary_json
        else output_root / "benchmark_qwen_offloading_v1_summary.json"
    )
    summary_md = (
        Path(args.summary_md)
        if args.summary_md
        else output_root / "benchmark_qwen_offloading_v1_summary.md"
    )
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(
        render_markdown_summary(benchmark_summary=summary),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
