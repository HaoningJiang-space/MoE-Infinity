from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from moe_infinity.utils.qwen_smoke import (
    dispatcher_stats_dict,
    make_fresh_offload_path,
    visible_cuda_devices,
)


QWEN_BENCHMARK_VARIANTS = (
    "on_demand",
    "trace_similarity_prefetch",
    "history_reuse_prefetch",
    "history_reuse_backbone",
)


@dataclass(frozen=True)
class QwenTraceRequest:
    request_id: str
    messages: List[Dict[str, str]]
    tag: str = ""


def build_qwen_benchmark_config(
    *,
    variant: str,
    offload_path: str,
    device_memory_ratio: float,
    num_threads: int,
    library_capacity: int,
    library_metric: str,
    library_admission: str,
    backbone_topk: int = 8,
    prefetch_future_layers: int = 4,
    prefetch_max_candidates: int = 32,
) -> Dict[str, object]:
    if variant not in QWEN_BENCHMARK_VARIANTS:
        raise ValueError(
            f"Unsupported benchmark variant '{variant}'. "
            f"Available: {QWEN_BENCHMARK_VARIANTS}"
        )

    if variant == "on_demand":
        offloading_policy = "baseline_trace_similarity"
        prefetch = False
        policy_score_only = False
        prefetch_backbone_topk = 0
        prefetch_future_layers = 0
        prefetch_max_candidates = 0
    elif variant == "trace_similarity_prefetch":
        offloading_policy = "baseline_trace_similarity"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = 0
    elif variant == "history_reuse_prefetch":
        offloading_policy = "finegrained_history_reuse"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = 0
    else:
        offloading_policy = "finegrained_history_reuse"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = int(backbone_topk)

    return {
        "offload_path": offload_path,
        "device_memory_ratio": device_memory_ratio,
        "num_threads": num_threads,
        "prefetch": prefetch,
        "policy_score_only": policy_score_only,
        "offloading_policy": offloading_policy,
        "historical_library_capacity": library_capacity,
        "historical_library_metric": library_metric,
        "historical_library_admission": library_admission,
        "prefetch_backbone_topk": prefetch_backbone_topk,
        "prefetch_future_layers": int(prefetch_future_layers),
        "prefetch_max_candidates": int(prefetch_max_candidates),
    }


def load_chat_trace(path: str | Path) -> List[QwenTraceRequest]:
    trace_path = Path(path)
    requests: List[QwenTraceRequest] = []
    for line_number, raw_line in enumerate(
        trace_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(
                f"{trace_path}:{line_number} missing non-empty request_id"
            )
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"{trace_path}:{line_number} missing non-empty messages list"
            )
        norm_messages: List[Dict[str, str]] = []
        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(
                    f"{trace_path}:{line_number} message[{idx}] must be an object"
                )
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError(
                    f"{trace_path}:{line_number} message[{idx}] must have string role/content"
                )
            norm_messages.append({"role": role, "content": content})
        tag = payload.get("tag", "")
        if tag is None:
            tag = ""
        if not isinstance(tag, str):
            raise ValueError(
                f"{trace_path}:{line_number} tag must be a string when present"
            )
        requests.append(
            QwenTraceRequest(
                request_id=request_id,
                messages=norm_messages,
                tag=tag,
            )
        )
    if not requests:
        raise ValueError(f"{trace_path} does not contain any requests")
    return requests


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def summarize_hit_rate_tensor(raw: torch.Tensor | None) -> Dict[str, float | int]:
    if raw is None:
        return {}
    if not isinstance(raw, torch.Tensor):
        raise TypeError(f"Expected hit-rate tensor, got {type(raw)!r}")
    if raw.numel() == 0:
        return {}
    matrix = raw.detach().cpu().to(torch.int64)
    if matrix.dim() != 2 or matrix.size(1) < 8:
        return {
            "num_rows": int(matrix.shape[0]) if matrix.dim() > 0 else 0,
            "num_cols": int(matrix.shape[1]) if matrix.dim() > 1 else 0,
        }

    visit_count = int(matrix[:, 0].sum().item())
    gpu_visit_count = int(matrix[:, 1].sum().item())
    cpu_visit_count = int(matrix[:, 2].sum().item())
    hit_count = int(matrix[:, 3].sum().item())
    gpu_hit_count = int(matrix[:, 4].sum().item())
    cpu_hit_count = int(matrix[:, 5].sum().item())
    prefetch_count = int(matrix[:, 7].sum().item())
    sparse_node_count = int((matrix[:, 10] > 0).sum().item()) if matrix.size(1) > 10 else 0

    def _safe_ratio(numer: int, denom: int) -> float:
        if denom <= 0:
            return 0.0
        return float(numer / denom)

    return {
        "visit_count": visit_count,
        "gpu_visit_count": gpu_visit_count,
        "cpu_visit_count": cpu_visit_count,
        "hit_count": hit_count,
        "gpu_hit_count": gpu_hit_count,
        "cpu_hit_count": cpu_hit_count,
        "prefetch_count": prefetch_count,
        "sparse_node_count": sparse_node_count,
        "overall_hit_rate": _safe_ratio(hit_count, visit_count),
        "gpu_hit_rate": _safe_ratio(gpu_hit_count, gpu_visit_count),
        "cpu_hit_rate": _safe_ratio(cpu_hit_count, cpu_visit_count),
    }


def diff_counter_dict(
    current: Mapping[str, int | float] | None,
    previous: Mapping[str, int | float] | None,
    *,
    keys: Iterable[str],
) -> Dict[str, int]:
    current = current or {}
    previous = previous or {}
    delta: Dict[str, int] = {}
    for key in keys:
        delta[key] = int(current.get(key, 0)) - int(previous.get(key, 0))
    return delta


def aggregate_request_records(
    request_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    latencies = [float(record["latency_s"]) for record in request_records]
    generated_tokens = [int(record["generated_tokens"]) for record in request_records]
    latency_per_token_ms = [
        (float(record["latency_s"]) * 1000.0) / max(int(record["generated_tokens"]), 1)
        for record in request_records
    ]
    total_latency_s = float(sum(latencies))
    total_generated_tokens = int(sum(generated_tokens))

    enqueue_counts = [
        int(record.get("dispatcher_stats", {}).get("enqueue_count", 0))
        for record in request_records
    ]
    busy_wait_counts = [
        int(record.get("dispatcher_stats", {}).get("busy_wait_count", 0))
        for record in request_records
    ]
    busy_wait_total_wait_us = [
        int(record.get("dispatcher_stats", {}).get("busy_wait_total_wait_us", 0))
        for record in request_records
    ]
    busy_wait_max_wait_us = [
        int(record.get("dispatcher_stats", {}).get("busy_wait_max_wait_us", 0))
        for record in request_records
    ]
    library_query_deltas = [
        int(record.get("library_stats_delta", {}).get("query_count", 0))
        for record in request_records
    ]
    library_hit_deltas = [
        int(record.get("library_stats_delta", {}).get("hit_count", 0))
        for record in request_records
    ]
    library_admit_deltas = [
        int(record.get("library_stats_delta", {}).get("admit_count", 0))
        for record in request_records
    ]
    cache_hit_rates = [
        float(record.get("cache_hit_rate_delta", {}).get("overall_hit_rate", 0.0))
        for record in request_records
        if "cache_hit_rate_delta" in record
    ]

    request_count = len(request_records)
    return {
        "request_count": request_count,
        "success_count": request_count,
        "failure_count": 0,
        "total_latency_s": total_latency_s,
        "generated_tokens_total": total_generated_tokens,
        "generated_tokens_per_second": (
            float(total_generated_tokens / total_latency_s)
            if total_latency_s > 0
            else 0.0
        ),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_per_generated_token_mean_ms": (
            float(sum(latency_per_token_ms) / request_count) if request_count else 0.0
        ),
        "latency_per_generated_token_p95_ms": percentile(
            latency_per_token_ms, 95
        ),
        "mean_dispatcher_enqueue_count": (
            float(sum(enqueue_counts) / request_count) if request_count else 0.0
        ),
        "mean_dispatcher_busy_wait_count": (
            float(sum(busy_wait_counts) / request_count) if request_count else 0.0
        ),
        "mean_dispatcher_busy_wait_total_wait_us": (
            float(sum(busy_wait_total_wait_us) / request_count)
            if request_count
            else 0.0
        ),
        "p95_dispatcher_busy_wait_max_wait_us": percentile(
            busy_wait_max_wait_us, 95
        ),
        "library_query_count_total": int(sum(library_query_deltas)),
        "library_hit_count_total": int(sum(library_hit_deltas)),
        "library_admit_count_total": int(sum(library_admit_deltas)),
        "library_query_count_mean": (
            float(sum(library_query_deltas) / request_count) if request_count else 0.0
        ),
        "library_hit_count_mean": (
            float(sum(library_hit_deltas) / request_count) if request_count else 0.0
        ),
        "library_admit_count_mean": (
            float(sum(library_admit_deltas) / request_count) if request_count else 0.0
        ),
        "mean_cache_hit_rate": (
            float(sum(cache_hit_rates) / len(cache_hit_rates))
            if cache_hit_rates
            else 0.0
        ),
    }


def discover_trace_files(trace_dir: str | Path) -> Dict[str, Path]:
    root = Path(trace_dir)
    traces = {}
    for path in sorted(root.glob("*.jsonl")):
        traces[path.stem] = path
    if not traces:
        raise ValueError(f"No JSONL traces found under {root}")
    return traces


def build_env_note(
    *,
    model_path: str,
    offload_root: str,
    visible_count: int,
    repo_root: str | Path,
) -> Dict[str, Any]:
    repo_root = Path(repo_root)

    def _git_output(args: Sequence[str]) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return ""
        return result.stdout.strip()

    head = _git_output(["rev-parse", "HEAD"])
    status = _git_output(["status", "--short"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "offload_root": offload_root,
        "visible_cuda": visible_cuda_devices(
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            names=(torch.cuda.get_device_name(i) for i in range(visible_count)),
        ),
        "git_head": head,
        "git_dirty": bool(status),
        "git_status_short": status.splitlines()[:50] if status else [],
    }


def render_markdown_summary(
    *,
    benchmark_summary: Mapping[str, Any],
) -> str:
    lines = [
        "# Qwen Offloading Benchmark v1",
        "",
        f"- Model: `{benchmark_summary['env']['model_path']}`",
        f"- Timestamp (UTC): `{benchmark_summary['env']['timestamp_utc']}`",
        f"- Variants: `{', '.join(benchmark_summary['variants'])}`",
        f"- Traces: `{', '.join(benchmark_summary['traces'])}`",
        "",
    ]
    trace_results = benchmark_summary["trace_results"]
    for trace_name in benchmark_summary["traces"]:
        lines.extend(
            [
                f"## Trace: `{trace_name}`",
                "",
                "| Variant | p50 latency (s) | p95 latency (s) | tok/s | mean enqueue | mean busy waits | mean cache hit rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        current = trace_results[trace_name]
        for variant in benchmark_summary["variants"]:
            agg = current[variant]["aggregate"]
            lines.append(
                "| {variant} | {p50:.4f} | {p95:.4f} | {tps:.3f} | {enqueue:.1f} | {busy:.1f} | {hit:.4f} |".format(
                    variant=variant,
                    p50=agg["latency_p50_s"],
                    p95=agg["latency_p95_s"],
                    tps=agg["generated_tokens_per_second"],
                    enqueue=agg["mean_dispatcher_enqueue_count"],
                    busy=agg["mean_dispatcher_busy_wait_count"],
                    hit=agg["mean_cache_hit_rate"],
                )
            )

        trace_prefetch = current.get("trace_similarity_prefetch", {}).get(
            "aggregate", {}
        ).get("generated_tokens_per_second", 0.0)
        history_prefetch = current.get("history_reuse_prefetch", {}).get(
            "aggregate", {}
        ).get("generated_tokens_per_second", 0.0)
        history_backbone = current.get("history_reuse_backbone", {}).get(
            "aggregate", {}
        ).get("generated_tokens_per_second", 0.0)
        history_help = (
            history_prefetch > trace_prefetch
            if "trace_similarity_prefetch" in current
            and "history_reuse_prefetch" in current
            else False
        )
        backbone_help = (
            history_backbone > history_prefetch
            if "history_reuse_prefetch" in current
            and "history_reuse_backbone" in current
            else False
        )
        recurrence_note = "n/a"
        if trace_name == "recurrence_heavy":
            recurrence_note = (
                "yes" if history_backbone > trace_prefetch else "no"
            )
        lines.extend(
            [
                "",
                f"- Conclusion: history reuse helps = `{str(history_help).lower()}`; backbone restriction helps = `{str(backbone_help).lower()}`; recurrence amplification = `{recurrence_note}`.",
                "",
            ]
        )
    return "\n".join(lines)


def prepare_case_paths(
    *,
    output_root: str | Path,
    trace_name: str,
    variant: str,
) -> Dict[str, str]:
    output_root = Path(output_root)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{trace_name}__{variant}"
    return {
        "raw_json": str(raw_dir / f"{slug}.json"),
        "offload_path": make_fresh_offload_path(
            str(output_root / "offload"), phase=slug
        ),
    }
