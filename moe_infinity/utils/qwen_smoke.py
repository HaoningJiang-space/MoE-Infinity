from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Iterable, Sequence


def build_qwen_smoke_config(
    *,
    phase: str,
    offload_path: str,
    device_memory_ratio: float,
    num_threads: int,
    library_capacity: int,
    library_metric: str,
    library_admission: str,
    backbone_topk: int,
) -> Dict[str, object]:
    if phase not in {"baseline", "history", "backbone", "backbone_score"}:
        raise ValueError(f"Unsupported smoke phase: {phase}")

    if phase == "baseline":
        offloading_policy = "baseline_trace_similarity"
        prefetch = False
        policy_score_only = False
        prefetch_backbone_topk = 0
        prefetch_backbone_mode = "raw_topk"
        prefetch_future_layers = 0
        prefetch_max_candidates = 0
    elif phase == "history":
        offloading_policy = "finegrained_history_reuse"
        prefetch = False
        policy_score_only = True
        prefetch_backbone_topk = 0
        prefetch_backbone_mode = "raw_topk"
        prefetch_future_layers = 0
        prefetch_max_candidates = 0
    elif phase == "backbone":
        offloading_policy = "finegrained_history_reuse"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = int(backbone_topk)
        prefetch_backbone_mode = "raw_topk"
        prefetch_future_layers = 4
        prefetch_max_candidates = 32
    else:
        offloading_policy = "finegrained_history_reuse"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = int(backbone_topk)
        prefetch_backbone_mode = "history_score"
        prefetch_future_layers = 4
        prefetch_max_candidates = 32

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
        "prefetch_backbone_mode": prefetch_backbone_mode,
        "prefetch_future_layers": prefetch_future_layers,
        "prefetch_max_candidates": prefetch_max_candidates,
        "prefetch_candidate_min_score": 1e-6,
    }


def make_fresh_offload_path(root: str, *, phase: str) -> str:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"{phase}_", dir=root_path)


def dispatcher_stats_dict(stats: Sequence[int] | None) -> Dict[str, int]:
    if not stats:
        return {}
    values = list(stats)
    if len(values) not in {4, 11, 18, 20, 24}:
        raise ValueError(f"Unexpected dispatcher stats payload: {values}")
    result = {
        "enqueue_count": int(values[0]),
        "busy_wait_count": int(values[1]),
        "busy_wait_total_wait_us": int(values[2]),
        "busy_wait_max_wait_us": int(values[3]),
    }
    if len(values) >= 11:
        result.update(
            {
                "cache_hit_fetch_count": int(values[4]),
                "cache_miss_fetch_count": int(values[5]),
                "eviction_count": int(values[6]),
                "all_locked_event_count": int(values[7]),
                "no_victim_wait_count": int(values[8]),
                "no_victim_wait_total_us": int(values[9]),
                "no_victim_wait_max_us": int(values[10]),
            }
        )
    if len(values) >= 18:
        result.update(
            {
                "fetch_dequeue_count": int(values[11]),
                "exec_dequeue_count": int(values[12]),
                "output_count": int(values[13]),
                "pending_wait_count": int(values[14]),
                "pending_wait_total_us": int(values[15]),
                "pending_wait_max_us": int(values[16]),
                "pending_stall_count": int(values[17]),
            }
        )
    if len(values) >= 20:
        result.update(
            {
                "prefetch_resident_hit_count": int(values[18]),
                "late_prefetch_demand_miss_count": int(values[19]),
            }
        )
    if len(values) >= 24:
        result.update(
            {
                "demand_candidate_protect_skip_count": int(values[20]),
                "demand_candidate_protect_fallback_count": int(values[21]),
                "candidate_resident_hit_count": int(values[22]),
                "candidate_demand_miss_count": int(values[23]),
            }
        )
    return result


def visible_cuda_devices(cuda_visible_devices: str | None, names: Iterable[str]) -> Dict[str, object]:
    return {
        "cuda_visible_devices": cuda_visible_devices,
        "visible_device_names": list(names),
    }
