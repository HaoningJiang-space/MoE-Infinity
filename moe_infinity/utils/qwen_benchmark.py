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
    "history_reuse_local_backbone",
    "history_reuse_topk_prefetch",
    "history_reuse_topk_backbone",
    "history_reuse_consensus_backbone",
    "history_reuse_consensus_backbone_retrieval",
    "history_reuse_backbone_score",
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
    prefetch_admission_enabled: bool = False,
    prefetch_admission_demand_reserve: int = 2,
    prefetch_admission_locked_ratio_threshold: float = 0.8,
    prefetch_admission_max_under_pressure: int = 4,
    prefetch_admission_max_per_plan: int = -1,
    prefetch_credit_gated_enabled: bool = False,
    prefetch_credit_count: int = -1,
    prefetch_credit_zero_action: str = "update_only",
    prefetch_policy_disabled: bool = False,
    historical_reuse_match_topk: int = 4,
    historical_reuse_match_min_required: int = 2,
    historical_reuse_consensus_min_votes: int = 2,
    historical_library_similarity_mode: str = "prefix_mean",
    historical_library_recent_window: int = 4,
    historical_library_recent_weight: float = 0.5,
    historical_reuse_object_mode: str = "sequence_matrix",
    local_continuation_library_capacity: int = 4096,
    local_continuation_key_layers: int = 4,
    local_continuation_future_layers: int = 4,
    local_continuation_match_topk: int = 4,
    local_continuation_match_min_required: int = 2,
    phasea_analysis_future_layers: int = 0,
    phasea_analysis_max_ranked_candidates: int = 128,
    policy_score_only_override: bool | None = None,
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
        prefetch_backbone_mode = "raw_topk"
        prefetch_future_layers = 0
        prefetch_max_candidates = 0
        historical_reuse_candidate_mode = "top1"
    elif variant == "trace_similarity_prefetch":
        offloading_policy = "baseline_trace_similarity"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = 0
        prefetch_backbone_mode = "raw_topk"
        historical_reuse_candidate_mode = "top1"
    elif variant == "history_reuse_prefetch":
        offloading_policy = "finegrained_history_reuse"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = 0
        prefetch_backbone_mode = "raw_topk"
        historical_reuse_candidate_mode = "top1"
    elif variant == "history_reuse_backbone":
        offloading_policy = "finegrained_history_reuse"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = int(backbone_topk)
        prefetch_backbone_mode = "raw_topk"
        historical_reuse_candidate_mode = "top1"
    elif variant == "history_reuse_local_backbone":
        offloading_policy = "finegrained_history_reuse"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = int(backbone_topk)
        prefetch_backbone_mode = "raw_topk"
        historical_reuse_candidate_mode = "top1"
        historical_reuse_object_mode = "local_continuation"
    elif variant == "history_reuse_topk_prefetch":
        offloading_policy = "finegrained_history_reuse"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = 0
        prefetch_backbone_mode = "raw_topk"
        historical_reuse_candidate_mode = "topk"
    elif variant == "history_reuse_topk_backbone":
        offloading_policy = "finegrained_history_reuse"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = int(backbone_topk)
        prefetch_backbone_mode = "raw_topk"
        historical_reuse_candidate_mode = "topk"
    elif variant == "history_reuse_consensus_backbone":
        offloading_policy = "finegrained_history_reuse"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = int(backbone_topk)
        prefetch_backbone_mode = "consensus_mask"
        historical_reuse_candidate_mode = "topk"
        historical_library_similarity_mode = "prefix_mean"
    elif variant == "history_reuse_consensus_backbone_retrieval":
        offloading_policy = "finegrained_history_reuse"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = int(backbone_topk)
        prefetch_backbone_mode = "consensus_mask"
        historical_reuse_candidate_mode = "topk"
        historical_library_similarity_mode = "prefix_recent_hybrid"
    else:
        offloading_policy = "finegrained_history_reuse"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = int(backbone_topk)
        prefetch_backbone_mode = "history_score"
        historical_reuse_candidate_mode = "top1"

    if policy_score_only_override is not None:
        policy_score_only = bool(policy_score_only_override)

    return {
        "offload_path": offload_path,
        "device_memory_ratio": device_memory_ratio,
        "num_threads": num_threads,
        "prefetch": prefetch,
        "policy_score_only": policy_score_only,
        "offloading_policy": offloading_policy,
        "historical_library_capacity": library_capacity,
        "historical_library_metric": library_metric,
        "historical_library_similarity_mode": historical_library_similarity_mode,
        "historical_library_recent_window": int(historical_library_recent_window),
        "historical_library_recent_weight": float(historical_library_recent_weight),
        "historical_library_admission": library_admission,
        "historical_reuse_object_mode": historical_reuse_object_mode,
        "historical_reuse_candidate_mode": historical_reuse_candidate_mode,
        "historical_reuse_match_topk": int(historical_reuse_match_topk),
        "historical_reuse_match_min_required": int(
            historical_reuse_match_min_required
        ),
        "historical_reuse_consensus_min_votes": int(
            historical_reuse_consensus_min_votes
        ),
        "prefetch_backbone_topk": prefetch_backbone_topk,
        "prefetch_backbone_mode": prefetch_backbone_mode,
        "prefetch_future_layers": int(prefetch_future_layers),
        "prefetch_max_candidates": int(prefetch_max_candidates),
        "prefetch_candidate_min_score": 1e-6,
        "prefetch_admission_enabled": bool(prefetch_admission_enabled),
        "prefetch_admission_demand_reserve": int(prefetch_admission_demand_reserve),
        "prefetch_admission_locked_ratio_threshold": float(
            prefetch_admission_locked_ratio_threshold
        ),
        "prefetch_admission_max_under_pressure": int(
            prefetch_admission_max_under_pressure
        ),
        "prefetch_admission_max_per_plan": int(
            prefetch_admission_max_per_plan
        ),
        "prefetch_credit_gated_enabled": bool(prefetch_credit_gated_enabled),
        "prefetch_credit_count": int(prefetch_credit_count),
        "prefetch_credit_zero_action": str(prefetch_credit_zero_action),
        "prefetch_policy_disabled": bool(prefetch_policy_disabled),
        "local_continuation_library_capacity": int(
            local_continuation_library_capacity
        ),
        "local_continuation_key_layers": int(local_continuation_key_layers),
        "local_continuation_future_layers": int(
            local_continuation_future_layers
        ),
        "local_continuation_match_topk": int(local_continuation_match_topk),
        "local_continuation_match_min_required": int(
            local_continuation_match_min_required
        ),
        "phasea_analysis_future_layers": int(phasea_analysis_future_layers),
        "phasea_analysis_max_ranked_candidates": int(
            phasea_analysis_max_ranked_candidates
        ),
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


def render_chat_prompt(tokenizer: object, messages: Sequence[Mapping[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )


def count_request_input_tokens(
    tokenizer: object,
    messages: Sequence[Mapping[str, str]],
) -> int:
    prompt = render_chat_prompt(tokenizer, messages)
    encoded = tokenizer(prompt, truncation=False)
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def validate_requests_within_token_budget(
    tokenizer: object,
    requests: Sequence[QwenTraceRequest],
    *,
    max_input_length: int,
    trace_name: str,
) -> Dict[str, float | int]:
    if max_input_length <= 0:
        raise ValueError("max_input_length must be positive")
    lengths: List[int] = []
    for request in requests:
        token_count = count_request_input_tokens(tokenizer, request.messages)
        if token_count > max_input_length:
            raise ValueError(
                f"Trace '{trace_name}' request '{request.request_id}' tokenized to "
                f"{token_count} tokens, exceeding max_input_length={max_input_length}"
            )
        lengths.append(token_count)
    if not lengths:
        return {
            "request_count": 0,
            "min_input_tokens": 0,
            "max_input_tokens": 0,
            "mean_input_tokens": 0.0,
        }
    return {
        "request_count": len(lengths),
        "min_input_tokens": min(lengths),
        "max_input_tokens": max(lengths),
        "mean_input_tokens": float(sum(lengths) / len(lengths)),
    }


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
    cache_hit_fetch_counts = [
        int(record.get("dispatcher_stats", {}).get("cache_hit_fetch_count", 0))
        for record in request_records
    ]
    cache_miss_fetch_counts = [
        int(record.get("dispatcher_stats", {}).get("cache_miss_fetch_count", 0))
        for record in request_records
    ]
    eviction_counts = [
        int(record.get("dispatcher_stats", {}).get("eviction_count", 0))
        for record in request_records
    ]
    all_locked_event_counts = [
        int(record.get("dispatcher_stats", {}).get("all_locked_event_count", 0))
        for record in request_records
    ]
    no_victim_wait_counts = [
        int(record.get("dispatcher_stats", {}).get("no_victim_wait_count", 0))
        for record in request_records
    ]
    no_victim_wait_total_us = [
        int(record.get("dispatcher_stats", {}).get("no_victim_wait_total_us", 0))
        for record in request_records
    ]
    no_victim_wait_max_us = [
        int(record.get("dispatcher_stats", {}).get("no_victim_wait_max_us", 0))
        for record in request_records
    ]
    fetch_dequeue_counts = [
        int(record.get("dispatcher_stats", {}).get("fetch_dequeue_count", 0))
        for record in request_records
    ]
    exec_dequeue_counts = [
        int(record.get("dispatcher_stats", {}).get("exec_dequeue_count", 0))
        for record in request_records
    ]
    output_counts = [
        int(record.get("dispatcher_stats", {}).get("output_count", 0))
        for record in request_records
    ]
    pending_wait_counts = [
        int(record.get("dispatcher_stats", {}).get("pending_wait_count", 0))
        for record in request_records
    ]
    pending_wait_total_us = [
        int(record.get("dispatcher_stats", {}).get("pending_wait_total_us", 0))
        for record in request_records
    ]
    pending_wait_max_us = [
        int(record.get("dispatcher_stats", {}).get("pending_wait_max_us", 0))
        for record in request_records
    ]
    pending_stall_counts = [
        int(record.get("dispatcher_stats", {}).get("pending_stall_count", 0))
        for record in request_records
    ]
    prefetch_candidate_counts = [
        int(record.get("prefetcher_stats", {}).get("prefetch_candidate_count", 0))
        for record in request_records
    ]
    prefetch_admitted_counts = [
        int(record.get("prefetcher_stats", {}).get("prefetch_admitted_count", 0))
        for record in request_records
    ]
    prefetch_enqueue_counts = [
        int(record.get("prefetcher_stats", {}).get("prefetch_enqueue_count", 0))
        for record in request_records
    ]
    prefetch_drop_counts = [
        int(record.get("prefetcher_stats", {}).get("prefetch_drop_count", 0))
        for record in request_records
    ]
    prefetch_drop_cap_counts = [
        int(record.get("prefetcher_stats", {}).get("prefetch_drop_cap_count", 0))
        for record in request_records
    ]
    prefetch_drop_pressure_counts = [
        int(record.get("prefetcher_stats", {}).get("prefetch_drop_pressure_count", 0))
        for record in request_records
    ]
    prefetch_drop_no_evictable_counts = [
        int(
            record.get("prefetcher_stats", {}).get(
                "prefetch_drop_no_evictable_count", 0
            )
        )
        for record in request_records
    ]
    demand_prefetch_conflict_counts = [
        int(record.get("prefetcher_stats", {}).get("demand_prefetch_conflict_count", 0))
        for record in request_records
    ]
    prefetch_under_pressure_counts = [
        int(record.get("prefetcher_stats", {}).get("prefetch_under_pressure_count", 0))
        for record in request_records
    ]
    prefetch_credit_skip_counts = [
        int(record.get("prefetcher_stats", {}).get("prefetch_credit_skip_count", 0))
        for record in request_records
    ]
    prefetch_credit_issued_totals = [
        int(record.get("prefetcher_stats", {}).get("prefetch_credit_issued_total", 0))
        for record in request_records
    ]
    prefetch_credit_limited_counts = [
        int(record.get("prefetcher_stats", {}).get("prefetch_credit_limited_count", 0))
        for record in request_records
    ]
    prefetch_credit_materialized_counts = [
        int(
            record.get("prefetcher_stats", {}).get(
                "prefetch_credit_materialized_count", 0
            )
        )
        for record in request_records
    ]
    prefetch_credit_skip_policy_update_counts = [
        int(
            record.get("prefetcher_stats", {}).get(
                "prefetch_credit_skip_policy_update_count", 0
            )
        )
        for record in request_records
    ]
    prefetch_plan_replace_counts = [
        int(record.get("prefetcher_stats", {}).get("prefetch_plan_replace_count", 0))
        for record in request_records
    ]
    prefetch_plan_empty_replace_counts = [
        int(
            record.get("prefetcher_stats", {}).get(
                "prefetch_plan_empty_replace_count", 0
            )
        )
        for record in request_records
    ]
    prefetch_plan_candidate_counts = [
        int(record.get("prefetcher_stats", {}).get("prefetch_plan_candidate_count", 0))
        for record in request_records
    ]
    prefetch_plan_cleared_candidate_counts = [
        int(
            record.get("prefetcher_stats", {}).get(
                "prefetch_plan_cleared_candidate_count", 0
            )
        )
        for record in request_records
    ]
    pressure_locked_max_values = [
        int(record.get("prefetcher_stats", {}).get("pressure_locked_max", 0))
        for record in request_records
    ]
    pressure_evictable_min_values = [
        int(record.get("prefetcher_stats", {}).get("pressure_evictable_min", 0))
        for record in request_records
        if int(record.get("prefetcher_stats", {}).get("pressure_sample_count", 0))
        > 0
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
    cache_prefetch_count_deltas = [
        int(record.get("cache_hit_rate_delta", {}).get("prefetch_count", 0))
        for record in request_records
    ]

    request_count = len(request_records)
    prefetch_candidate_count_total = int(sum(prefetch_candidate_counts))
    prefetch_admitted_count_total = int(sum(prefetch_admitted_counts))
    prefetch_drop_cap_count_total = int(sum(prefetch_drop_cap_counts))
    prefetch_drop_pressure_count_total = int(sum(prefetch_drop_pressure_counts))
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
        "dispatcher_cache_hit_fetch_count_total": int(sum(cache_hit_fetch_counts)),
        "dispatcher_cache_miss_fetch_count_total": int(sum(cache_miss_fetch_counts)),
        "dispatcher_eviction_count_total": int(sum(eviction_counts)),
        "dispatcher_all_locked_event_count_total": int(sum(all_locked_event_counts)),
        "dispatcher_no_victim_wait_count_total": int(sum(no_victim_wait_counts)),
        "dispatcher_no_victim_wait_total_us": int(sum(no_victim_wait_total_us)),
        "dispatcher_fetch_dequeue_count_total": int(sum(fetch_dequeue_counts)),
        "dispatcher_exec_dequeue_count_total": int(sum(exec_dequeue_counts)),
        "dispatcher_output_count_total": int(sum(output_counts)),
        "dispatcher_pending_wait_count_total": int(sum(pending_wait_counts)),
        "dispatcher_pending_wait_total_us": int(sum(pending_wait_total_us)),
        "dispatcher_pending_stall_count_total": int(sum(pending_stall_counts)),
        "prefetch_candidate_count_total": prefetch_candidate_count_total,
        "prefetch_admitted_count_total": prefetch_admitted_count_total,
        "prefetch_enqueue_count_total": int(sum(prefetch_enqueue_counts)),
        "prefetch_drop_count_total": int(sum(prefetch_drop_counts)),
        "prefetch_drop_cap_count_total": prefetch_drop_cap_count_total,
        "prefetch_drop_pressure_count_total": prefetch_drop_pressure_count_total,
        "prefetch_drop_no_evictable_count_total": int(
            sum(prefetch_drop_no_evictable_counts)
        ),
        "demand_prefetch_conflict_count_total": int(
            sum(demand_prefetch_conflict_counts)
        ),
        "prefetch_under_pressure_count_total": int(
            sum(prefetch_under_pressure_counts)
        ),
        "prefetch_credit_skip_count_total": int(sum(prefetch_credit_skip_counts)),
        "prefetch_credit_issued_total": int(sum(prefetch_credit_issued_totals)),
        "prefetch_credit_limited_count_total": int(
            sum(prefetch_credit_limited_counts)
        ),
        "prefetch_credit_materialized_count_total": int(
            sum(prefetch_credit_materialized_counts)
        ),
        "prefetch_credit_skip_policy_update_count_total": int(
            sum(prefetch_credit_skip_policy_update_counts)
        ),
        "prefetch_plan_replace_count_total": int(sum(prefetch_plan_replace_counts)),
        "prefetch_plan_empty_replace_count_total": int(
            sum(prefetch_plan_empty_replace_counts)
        ),
        "prefetch_plan_candidate_count_total": int(
            sum(prefetch_plan_candidate_counts)
        ),
        "prefetch_plan_cleared_candidate_count_total": int(
            sum(prefetch_plan_cleared_candidate_counts)
        ),
        "prefetch_admit_rate": (
            float(prefetch_admitted_count_total / prefetch_candidate_count_total)
            if prefetch_candidate_count_total
            else 0.0
        ),
        "prefetch_pressure_drop_rate": (
            float(prefetch_drop_pressure_count_total / prefetch_candidate_count_total)
            if prefetch_candidate_count_total
            else 0.0
        ),
        "pressure_locked_max": (
            int(max(pressure_locked_max_values)) if pressure_locked_max_values else 0
        ),
        "pressure_evictable_min": (
            int(min(pressure_evictable_min_values))
            if pressure_evictable_min_values
            else 0
        ),
        "mean_dispatcher_cache_hit_fetch_count": (
            float(sum(cache_hit_fetch_counts) / request_count)
            if request_count
            else 0.0
        ),
        "mean_dispatcher_cache_miss_fetch_count": (
            float(sum(cache_miss_fetch_counts) / request_count)
            if request_count
            else 0.0
        ),
        "mean_dispatcher_eviction_count": (
            float(sum(eviction_counts) / request_count) if request_count else 0.0
        ),
        "mean_dispatcher_all_locked_event_count": (
            float(sum(all_locked_event_counts) / request_count)
            if request_count
            else 0.0
        ),
        "mean_dispatcher_no_victim_wait_count": (
            float(sum(no_victim_wait_counts) / request_count)
            if request_count
            else 0.0
        ),
        "mean_dispatcher_no_victim_wait_total_us": (
            float(sum(no_victim_wait_total_us) / request_count)
            if request_count
            else 0.0
        ),
        "p95_dispatcher_no_victim_wait_max_us": percentile(
            no_victim_wait_max_us, 95
        ),
        "mean_dispatcher_fetch_dequeue_count": (
            float(sum(fetch_dequeue_counts) / request_count) if request_count else 0.0
        ),
        "mean_dispatcher_exec_dequeue_count": (
            float(sum(exec_dequeue_counts) / request_count) if request_count else 0.0
        ),
        "mean_dispatcher_output_count": (
            float(sum(output_counts) / request_count) if request_count else 0.0
        ),
        "mean_dispatcher_pending_wait_count": (
            float(sum(pending_wait_counts) / request_count) if request_count else 0.0
        ),
        "mean_dispatcher_pending_wait_total_us": (
            float(sum(pending_wait_total_us) / request_count)
            if request_count
            else 0.0
        ),
        "p95_dispatcher_pending_wait_max_us": percentile(
            pending_wait_max_us, 95
        ),
        "mean_dispatcher_pending_stall_count": (
            float(sum(pending_stall_counts) / request_count)
            if request_count
            else 0.0
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
        "cache_prefetch_count_total": int(sum(cache_prefetch_count_deltas)),
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
                "| Variant | p50 latency (s) | p95 latency (s) | tok/s | mean enqueue | mean busy waits | mean evictions | mean all-locked | mean no-victim wait us | mean pending wait us | mean pending stalls | prefetch drop | cap drop | pressure drop | credit skip | credit issued | credit materialized | plan replace | empty replace | completed prefetch | admit rate | pressure drop rate | under pressure | conflict | locked max | evict min | mean cache hit rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        current = trace_results[trace_name]
        for variant in benchmark_summary["variants"]:
            agg = current[variant]["aggregate"]
            lines.append(
                "| {variant} | {p50:.4f} | {p95:.4f} | {tps:.3f} | {enqueue:.1f} | {busy:.1f} | {evict:.1f} | {all_locked:.2f} | {no_victim_us:.1f} | {pending_wait_us:.1f} | {pending_stalls:.2f} | {prefetch_drop} | {cap_drop} | {pressure_drop} | {credit_skip} | {credit_issued} | {credit_materialized} | {plan_replace} | {empty_replace} | {completed_prefetch} | {admit_rate:.4f} | {pressure_drop_rate:.4f} | {under_pressure} | {conflict} | {locked_max} | {evict_min} | {hit:.4f} |".format(
                    variant=variant,
                    p50=agg["latency_p50_s"],
                    p95=agg["latency_p95_s"],
                    tps=agg["generated_tokens_per_second"],
                    enqueue=agg["mean_dispatcher_enqueue_count"],
                    busy=agg["mean_dispatcher_busy_wait_count"],
                    evict=agg.get("mean_dispatcher_eviction_count", 0.0),
                    all_locked=agg.get(
                        "mean_dispatcher_all_locked_event_count", 0.0
                    ),
                    no_victim_us=agg.get(
                        "mean_dispatcher_no_victim_wait_total_us", 0.0
                    ),
                    pending_wait_us=agg.get(
                        "mean_dispatcher_pending_wait_total_us", 0.0
                    ),
                    pending_stalls=agg.get(
                        "mean_dispatcher_pending_stall_count", 0.0
                    ),
                    prefetch_drop=agg.get("prefetch_drop_count_total", 0),
                    cap_drop=agg.get("prefetch_drop_cap_count_total", 0),
                    pressure_drop=agg.get("prefetch_drop_pressure_count_total", 0),
                    credit_skip=agg.get("prefetch_credit_skip_count_total", 0),
                    credit_issued=agg.get("prefetch_credit_issued_total", 0),
                    credit_materialized=agg.get(
                        "prefetch_credit_materialized_count_total",
                        0,
                    ),
                    plan_replace=agg.get("prefetch_plan_replace_count_total", 0),
                    empty_replace=agg.get(
                        "prefetch_plan_empty_replace_count_total",
                        0,
                    ),
                    completed_prefetch=agg.get("cache_prefetch_count_total", 0),
                    admit_rate=agg.get("prefetch_admit_rate", 0.0),
                    pressure_drop_rate=agg.get("prefetch_pressure_drop_rate", 0.0),
                    under_pressure=agg.get(
                        "prefetch_under_pressure_count_total",
                        0,
                    ),
                    conflict=agg.get("demand_prefetch_conflict_count_total", 0),
                    locked_max=agg.get("pressure_locked_max", 0),
                    evict_min=agg.get("pressure_evictable_min", 0),
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
    events_dir = output_root / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{trace_name}__{variant}"
    return {
        "raw_json": str(raw_dir / f"{slug}.json"),
        "phasea_events_jsonl": str(events_dir / f"{slug}.jsonl"),
        "offload_path": make_fresh_offload_path(
            str(output_root / "offload"), phase=slug
        ),
    }
