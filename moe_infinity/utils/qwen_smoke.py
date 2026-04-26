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
    if phase not in {"baseline", "history", "backbone"}:
        raise ValueError(f"Unsupported smoke phase: {phase}")

    if phase == "baseline":
        offloading_policy = "baseline_trace_similarity"
        prefetch = False
        policy_score_only = False
        prefetch_backbone_topk = 0
        prefetch_future_layers = 0
        prefetch_max_candidates = 0
    elif phase == "history":
        offloading_policy = "finegrained_history_reuse"
        prefetch = False
        policy_score_only = True
        prefetch_backbone_topk = 0
        prefetch_future_layers = 0
        prefetch_max_candidates = 0
    else:
        offloading_policy = "finegrained_history_reuse"
        prefetch = True
        policy_score_only = False
        prefetch_backbone_topk = int(backbone_topk)
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
        "prefetch_future_layers": prefetch_future_layers,
        "prefetch_max_candidates": prefetch_max_candidates,
    }


def make_fresh_offload_path(root: str, *, phase: str) -> str:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"{phase}_", dir=root_path)


def dispatcher_stats_dict(stats: Sequence[int] | None) -> Dict[str, int]:
    if not stats:
        return {}
    values = list(stats)
    if len(values) != 4:
        raise ValueError(f"Unexpected dispatcher stats payload: {values}")
    return {
        "enqueue_count": int(values[0]),
        "busy_wait_count": int(values[1]),
        "busy_wait_total_wait_us": int(values[2]),
        "busy_wait_max_wait_us": int(values[3]),
    }


def visible_cuda_devices(cuda_visible_devices: str | None, names: Iterable[str]) -> Dict[str, object]:
    return {
        "cuda_visible_devices": cuda_visible_devices,
        "visible_device_names": list(names),
    }
