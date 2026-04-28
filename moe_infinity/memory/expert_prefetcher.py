# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team


import numpy as np
from transformers import PretrainedConfig

from moe_infinity.utils.prefetch_plan import rank_prefetch_candidates
from moe_infinity.utils import parse_moe_param


class ExpertPrefetcher(object):
    cache_file_rd = None
    first_k_dense_replace: int = 0

    def __init__(self, config: PretrainedConfig):
        print(config)
        self.num_layers, self.num_experts, self.num_encoder_layers = (
            parse_moe_param(config)
        )
        self.prefetch_admission_enabled = False
        self.prefetch_admission_demand_reserve = 2
        self.prefetch_admission_locked_ratio_threshold = 0.8
        self.prefetch_admission_max_under_pressure = 4
        self.prefetch_admission_max_per_plan = -1
        self.prefetch_credit_gated_enabled = False
        self.prefetch_credit_count = -1
        self.prefetch_credit_zero_action = "update_only"
        self.prefetch_policy_disabled = False
        self.reset_prefetch_runtime_stats()

    def set_archer_engine(self, archer_engine):
        global _expert_prefetcher
        _expert_prefetcher = archer_engine
        self.archer_engine = archer_engine

    def reset_prefetch_runtime_stats(self):
        self._prefetch_runtime_stats = {
            "prefetch_candidate_count": 0,
            "prefetch_admitted_count": 0,
            "prefetch_enqueue_count": 0,
            "prefetch_drop_count": 0,
            "prefetch_drop_cap_count": 0,
            "prefetch_drop_pressure_count": 0,
            "prefetch_drop_no_evictable_count": 0,
            "prefetch_under_pressure_count": 0,
            "demand_prefetch_conflict_count": 0,
            "pressure_sample_count": 0,
            "pressure_locked_max": 0,
            "pressure_evictable_min": None,
            "prefetch_credit_skip_count": 0,
            "prefetch_credit_issued_total": 0,
            "prefetch_credit_limited_count": 0,
            "prefetch_credit_materialized_count": 0,
            "prefetch_credit_skip_policy_update_count": 0,
            "prefetch_plan_replace_count": 0,
            "prefetch_plan_empty_replace_count": 0,
            "prefetch_plan_candidate_count": 0,
            "prefetch_plan_cleared_candidate_count": 0,
        }
        self._last_prefetch_plan_candidate_count = 0
        archer_engine = getattr(self, "archer_engine", None)
        if archer_engine is not None and hasattr(
            archer_engine, "reset_prefetch_lifecycle_stats"
        ):
            try:
                archer_engine.reset_prefetch_lifecycle_stats()
            except Exception:
                pass

    def prefetch_runtime_stats(self):
        stats = dict(self._prefetch_runtime_stats)
        if stats["pressure_evictable_min"] is None:
            stats["pressure_evictable_min"] = 0
        archer_engine = getattr(self, "archer_engine", None)
        if archer_engine is not None and hasattr(
            archer_engine, "get_prefetch_lifecycle_stats"
        ):
            try:
                values = list(archer_engine.get_prefetch_lifecycle_stats())
            except Exception:
                values = []
            names = [
                "prefetch_runtime_plan_replace_count",
                "prefetch_runtime_plan_empty_replace_count",
                "prefetch_runtime_plan_candidate_count",
                "prefetch_runtime_candidate_set_cleared_count",
                "prefetch_runtime_queue_cleared_task_count",
                "prefetch_runtime_enqueue_count",
                "prefetch_runtime_queue_push_count",
                "prefetch_runtime_same_device_skip_count",
                "prefetch_runtime_dequeue_count",
                "prefetch_runtime_complete_count",
                "prefetch_runtime_trylock_failed_count",
                "prefetch_runtime_evict_failed_count",
            ]
            for index, name in enumerate(names):
                stats[name] = int(values[index]) if index < len(values) else 0
        return stats

    def _pressure_snapshot(self, gpu_id):
        if not hasattr(self.archer_engine, "get_sparse_pressure_snapshot"):
            return None
        try:
            values = list(self.archer_engine.get_sparse_pressure_snapshot(gpu_id))
        except Exception:
            return None
        if len(values) < 6:
            return None
        return {
            "cached": int(values[0]),
            "locked": int(values[1]),
            "evictable": int(values[2]),
            "sparse_bytes": int(values[3]),
            "sparse_cache_limit": int(values[4]),
            "free_memory": int(values[5]),
        }

    def _record_pressure_snapshot(self, snapshot):
        if snapshot is None:
            return
        stats = self._prefetch_runtime_stats
        stats["pressure_sample_count"] += 1
        stats["pressure_locked_max"] = max(
            int(stats["pressure_locked_max"]), int(snapshot["locked"])
        )
        current_min = stats["pressure_evictable_min"]
        evictable = int(snapshot["evictable"])
        stats["pressure_evictable_min"] = (
            evictable if current_min is None else min(int(current_min), evictable)
        )

    def _admit_prefetch_tensor_ids(self, tensor_ids):
        stats = self._prefetch_runtime_stats
        stats["prefetch_candidate_count"] += len(tensor_ids)
        if not tensor_ids:
            return []
        if not bool(getattr(self, "prefetch_admission_enabled", False)):
            stats["prefetch_admitted_count"] += len(tensor_ids)
            return tensor_ids

        demand_reserve = max(
            int(getattr(self, "prefetch_admission_demand_reserve", 2)), 0
        )
        locked_ratio_threshold = float(
            getattr(self, "prefetch_admission_locked_ratio_threshold", 0.8)
        )
        max_under_pressure = max(
            int(getattr(self, "prefetch_admission_max_under_pressure", 4)), 0
        )
        max_per_plan = int(getattr(self, "prefetch_admission_max_per_plan", -1))

        admitted = []
        admitted_by_gpu = {}
        snapshots = {}
        conflict_gpus = set()
        for tensor_id in tensor_ids:
            gpu_id = int(self.archer_engine.get_node_default_device([tensor_id]))
            if gpu_id not in snapshots:
                snapshots[gpu_id] = self._pressure_snapshot(gpu_id)
                self._record_pressure_snapshot(snapshots[gpu_id])
            snapshot = snapshots[gpu_id]
            if snapshot is None:
                if max_per_plan >= 0 and admitted_by_gpu.get(gpu_id, 0) >= max_per_plan:
                    stats["prefetch_drop_count"] += 1
                    stats["prefetch_drop_cap_count"] += 1
                    continue
                admitted.append(tensor_id)
                admitted_by_gpu[gpu_id] = admitted_by_gpu.get(gpu_id, 0) + 1
                continue

            cached = int(snapshot["cached"])
            locked = int(snapshot["locked"])
            evictable = int(snapshot["evictable"])
            locked_ratio = (locked / cached) if cached > 0 else 0.0
            under_pressure = evictable <= demand_reserve or (
                cached > 0 and locked_ratio >= locked_ratio_threshold
            )
            if under_pressure:
                stats["prefetch_under_pressure_count"] += 1

            if evictable <= demand_reserve:
                stats["prefetch_drop_count"] += 1
                stats["prefetch_drop_pressure_count"] += 1
                stats["prefetch_drop_no_evictable_count"] += 1
                if gpu_id not in conflict_gpus:
                    stats["demand_prefetch_conflict_count"] += 1
                    conflict_gpus.add(gpu_id)
                continue

            if max_per_plan >= 0 and admitted_by_gpu.get(gpu_id, 0) >= max_per_plan:
                stats["prefetch_drop_count"] += 1
                stats["prefetch_drop_cap_count"] += 1
                continue

            if (
                cached > 0
                and locked_ratio >= locked_ratio_threshold
                and admitted_by_gpu.get(gpu_id, 0) >= max_under_pressure
            ):
                stats["prefetch_drop_count"] += 1
                stats["prefetch_drop_pressure_count"] += 1
                continue

            admitted.append(tensor_id)
            admitted_by_gpu[gpu_id] = admitted_by_gpu.get(gpu_id, 0) + 1

        stats["prefetch_admitted_count"] += len(admitted)
        return admitted

    def speculation_credit(self):
        if not bool(getattr(self, "prefetch_credit_gated_enabled", False)):
            return -1
        return max(int(getattr(self, "prefetch_credit_count", 0)), 0)

    def record_credit_skip(self):
        stats = self._prefetch_runtime_stats
        stats["prefetch_credit_skip_count"] += 1

    def record_credit_skip_policy_update(self):
        stats = self._prefetch_runtime_stats
        stats["prefetch_credit_skip_policy_update_count"] += 1

    def _replace_prefetch_plan(self, tensor_ids):
        stats = self._prefetch_runtime_stats
        candidate_count = len(tensor_ids)
        stats["prefetch_plan_replace_count"] += 1
        stats["prefetch_plan_candidate_count"] += candidate_count
        stats["prefetch_plan_cleared_candidate_count"] += int(
            getattr(self, "_last_prefetch_plan_candidate_count", 0)
        )
        if candidate_count == 0:
            stats["prefetch_plan_empty_replace_count"] += 1
        self._last_prefetch_plan_candidate_count = candidate_count
        self.archer_engine.replace_cache_candidates(tensor_ids)

    def prefetch_experts_list(self, layer_id, expert_list):
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(self.expert_tensor_map[(layer_id, j)])
        admitted_tensor_ids = self._admit_prefetch_tensor_ids(tensor_ids)
        if bool(getattr(self, "prefetch_admission_enabled", False)):
            self._replace_prefetch_plan(admitted_tensor_ids)
        for tensor_id in admitted_tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)
            self._prefetch_runtime_stats["prefetch_enqueue_count"] += 1

    def fetch_experts_lock_cache(self, layer_id, expert_list):
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(self.expert_tensor_map[(layer_id, j)])
        self.archer_engine.replace_cache_candidates(tensor_ids)

    def _effective_max_candidates(self, max_candidates, max_candidates_override):
        if max_candidates_override is None or int(max_candidates_override) < 0:
            return int(max_candidates)
        override = max(int(max_candidates_override), 0)
        if int(max_candidates) > 0:
            return min(int(max_candidates), override)
        return override

    def prefetch_experts(self, layer_id, expert_matrix, max_candidates_override=None):
        future_layers = int(getattr(self, "prefetch_future_layers", 0))
        max_candidates = int(getattr(self, "prefetch_max_candidates", 0))
        effective_max_candidates = self._effective_max_candidates(
            max_candidates,
            max_candidates_override,
        )
        min_score = float(getattr(self, "prefetch_candidate_min_score", 1e-6))
        stats = self._prefetch_runtime_stats
        if (
            max_candidates_override is not None
            and int(max_candidates_override) >= 0
        ):
            credit = max(int(max_candidates_override), 0)
            stats["prefetch_credit_issued_total"] += credit
            if max_candidates <= 0 or credit < max_candidates:
                stats["prefetch_credit_limited_count"] += 1
            if credit == 0:
                return
        ranked_candidates = rank_prefetch_candidates(
            layer_id=layer_id,
            expert_matrix=expert_matrix,
            future_layers=future_layers,
            max_candidates=effective_max_candidates,
            min_score=min_score,
        )
        if (
            max_candidates_override is not None
            and int(max_candidates_override) >= 0
        ):
            stats["prefetch_credit_materialized_count"] += len(ranked_candidates)
        tensor_ids = [
            self.expert_tensor_map[(candidate.layer_idx, candidate.expert_idx)]
            for candidate in ranked_candidates
        ]
        assert len(np.unique(tensor_ids)) == len(tensor_ids)
        admitted_tensor_ids = self._admit_prefetch_tensor_ids(tensor_ids)
        self._replace_prefetch_plan(admitted_tensor_ids)
        for tensor_id in admitted_tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)
            self._prefetch_runtime_stats["prefetch_enqueue_count"] += 1
