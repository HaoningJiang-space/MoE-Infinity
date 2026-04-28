from __future__ import annotations

from collections import Counter
from typing import Dict, Optional

import numpy as np
import torch

from moe_infinity.analysis.phasea import PhaseAObservationRecorder
from moe_infinity.utils.prefetch_plan import count_prefetch_candidates

from .base import BackboneSupport
from .history_reuse import FineGrainedHistoryReusePolicy
from .historical_library import HistoricalExpertLibrary
from .local_continuation_library import LocalContinuationLibrary
from .static_hot import StaticHotPrefetchPolicy
from .static_frequency import StaticFrequencyPolicy
from .trace_similarity import TraceSimilarityPolicy


class OffloadingPolicyManager:
    def __init__(self, *, config, tracer, predictor, model_tag: str = "") -> None:
        self.config = config
        self.tracer = tracer
        self.predictor = predictor
        self.model_tag = model_tag
        self.library = HistoricalExpertLibrary(
            capacity=config.historical_library_capacity,
            metric=config.historical_library_metric,
            similarity_mode=getattr(
                config,
                "historical_library_similarity_mode",
                "prefix_mean",
            ),
            recent_window=getattr(
                config,
                "historical_library_recent_window",
                4,
            ),
            recent_weight=getattr(
                config,
                "historical_library_recent_weight",
                0.5,
            ),
            admission_policy=config.historical_library_admission,
            dedup_threshold=config.historical_library_dedup_threshold,
        )
        self.local_library = LocalContinuationLibrary(
            capacity=getattr(config, "local_continuation_library_capacity", 4096),
            key_layers=getattr(config, "local_continuation_key_layers", 4),
            future_layers=getattr(config, "local_continuation_future_layers", 4),
            metric=config.historical_library_metric,
            admission_policy=config.historical_library_admission,
            dedup_threshold=config.historical_library_dedup_threshold,
            value_threshold=getattr(config, "prefetch_candidate_min_score", 1e-6),
        )
        self.policies: Dict[str, object] = {
            "baseline_trace_similarity": TraceSimilarityPolicy(
                config=config,
                tracer=tracer,
                predictor=predictor,
                model_tag=model_tag,
            ),
            "finegrained_history_reuse": FineGrainedHistoryReusePolicy(
                config=config,
                tracer=tracer,
                predictor=predictor,
                library=self.library,
                local_library=self.local_library,
                model_tag=model_tag,
            ),
            "static_frequency": StaticFrequencyPolicy(
                config=config,
                tracer=tracer,
                predictor=predictor,
                model_tag=model_tag,
            ),
            "static_hot_prefetch": StaticHotPrefetchPolicy(
                config=config,
                tracer=tracer,
                predictor=predictor,
                model_tag=model_tag,
            ),
        }
        self.policy_name = config.offloading_policy
        if self.policy_name not in self.policies:
            raise ValueError(
                f"Unsupported offloading policy '{self.policy_name}'. "
                f"Available: {sorted(self.policies)}"
            )
        self.phasea_recorder: Optional[PhaseAObservationRecorder] = None

    @property
    def policy(self):
        return self.policies[self.policy_name]

    def attach_phasea_recorder(
        self, recorder: Optional[PhaseAObservationRecorder]
    ) -> None:
        self.phasea_recorder = recorder

    def _project_backbone(self, matrix: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if matrix is None:
            return None
        topk = int(getattr(self.config, "prefetch_backbone_topk", 0))
        if topk <= 0:
            return matrix
        projected = np.zeros_like(matrix)
        for layer_idx in range(matrix.shape[0]):
            row = matrix[layer_idx]
            positive = np.flatnonzero(row > 0)
            if positive.size <= topk:
                projected[layer_idx] = row
                continue
            keep = positive[np.argsort(row[positive])[-topk:]]
            projected[layer_idx, keep] = row[keep]
        return projected

    def _project_backbone_with_scores(
        self,
        *,
        base_matrix: np.ndarray,
        score_matrix: np.ndarray,
    ) -> np.ndarray:
        topk = int(getattr(self.config, "prefetch_backbone_topk", 0))
        if topk <= 0:
            return np.asarray(base_matrix, dtype=np.float32).copy()
        projected = np.zeros_like(base_matrix)
        for layer_idx in range(base_matrix.shape[0]):
            base_row = base_matrix[layer_idx]
            score_row = score_matrix[layer_idx]
            positive = np.flatnonzero(base_row > 0)
            if positive.size == 0:
                continue
            if positive.size <= topk:
                projected[layer_idx, positive] = base_row[positive]
                continue
            keep = positive[np.argsort(score_row[positive])[-topk:]]
            projected[layer_idx, keep] = base_row[keep]
        return projected

    def _compute_history_backbone_score_matrix(
        self,
        supports: list[BackboneSupport],
    ) -> np.ndarray:
        matrices = [
            np.asarray(support.matrix, dtype=np.float32) for support in supports
        ]
        if not matrices:
            raise ValueError("At least one backbone support is required.")
        weights = np.asarray(
            [max(float(support.weight), 0.0) for support in supports],
            dtype=np.float32,
        )
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            weights = np.zeros_like(weights)
            weights[0] = 1.0
        else:
            weights = weights / weight_sum
        stacked = np.stack(matrices, axis=0)
        importance = np.tensordot(weights, stacked, axes=(0, 0)).astype(np.float32)
        centered = stacked - importance[None, ...]
        variance = np.tensordot(weights, centered * centered, axes=(0, 0))
        instability = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
        penalty = float(getattr(self.config, "prefetch_backbone_lambda", 0.5))
        return importance - penalty * instability

    def _project_prefetch_matrix(
        self,
        *,
        seq_id: str,
        layer_idx: int,
        base_matrix: Optional[np.ndarray],
        score_metadata: Optional[Dict[str, int | str | bool]] = None,
    ) -> tuple[Optional[np.ndarray], Dict[str, int | str | bool | float]]:
        metadata: Dict[str, int | str | bool | float] = {
            "candidate_source": "empty",
            "fallback_reason": "none",
            "library_match_count": 0,
            "local_match_count": 0,
            "raw_positive_count": 0,
            "kept_candidate_count": 0,
            "all_scores_below_threshold": False,
            "backbone_mode": "disabled",
            "backbone_support_count": 0,
            "backbone_history_match_count": 0,
            "backbone_projection_topk": int(
                getattr(self.config, "prefetch_backbone_topk", 0)
            ),
            "backbone_fallback_used": False,
            "consensus_min_votes": int(
                getattr(self.config, "historical_reuse_consensus_min_votes", 1)
            ),
            "consensus_support_count": 0,
            "consensus_retained_count": 0,
            "consensus_fallback_used": False,
            "consensus_fallback_reason": "none",
            "historical_reuse_object_mode": getattr(
                self.config,
                "historical_reuse_object_mode",
                "sequence_matrix",
            ),
            "local_continuation_key_layers": int(
                getattr(self.config, "local_continuation_key_layers", 4)
            ),
            "local_continuation_future_layers": int(
                getattr(self.config, "local_continuation_future_layers", 4)
            ),
        }
        if score_metadata:
            metadata.update(score_metadata)
        if base_matrix is None:
            return None, metadata

        min_score = float(getattr(self.config, "prefetch_candidate_min_score", 1e-6))
        future_layers = int(getattr(self.config, "prefetch_future_layers", 0))
        raw_positive_count = count_prefetch_candidates(
            layer_id=layer_idx,
            expert_matrix=base_matrix,
            future_layers=future_layers,
            min_score=min_score,
        )
        metadata["raw_positive_count"] = int(raw_positive_count)
        metadata["all_scores_below_threshold"] = bool(raw_positive_count == 0)

        topk = int(getattr(self.config, "prefetch_backbone_topk", 0))
        if topk <= 0:
            metadata["kept_candidate_count"] = int(raw_positive_count)
            if raw_positive_count == 0:
                metadata["candidate_source"] = "empty"
                metadata["fallback_reason"] = "all_scores_below_threshold"
            return base_matrix, metadata

        configured_mode = str(
            getattr(self.config, "prefetch_backbone_mode", "raw_topk")
        )
        metadata["backbone_mode"] = configured_mode
        if (
            metadata.get("historical_reuse_object_mode", "sequence_matrix")
            != "local_continuation"
            and metadata.get("candidate_source") in {
            "history_match",
            "history_match_topk",
            "history_match_top1_fallback",
            "history_fallback_current_trace",
        }
        ):
            if configured_mode == "history_score":
                metadata["candidate_source"] = "backbone_history_score_projection"
            elif configured_mode == "consensus_mask":
                metadata["candidate_source"] = "backbone_consensus_projection"
            else:
                metadata["candidate_source"] = "backbone_raw_projection"
        if configured_mode == "consensus_mask" and self.policy_name == "finegrained_history_reuse":
            consensus_mask, consensus_metadata = self.policy.consensus_backbone_mask(
                seq_id=seq_id,
                layer_idx=layer_idx,
                min_score=min_score,
            )
            metadata.update(consensus_metadata)
            if consensus_mask is None:
                metadata["consensus_fallback_used"] = True
                metadata["consensus_fallback_reason"] = str(
                    consensus_metadata.get("consensus_fallback_reason", "library_miss")
                )
                metadata["backbone_fallback_used"] = True
                metadata["fallback_reason"] = str(
                    consensus_metadata.get("consensus_fallback_reason", "library_miss")
                )
                metadata["candidate_source"] = "backbone_raw_projection"
                projected = self._project_backbone(base_matrix)
                kept_candidate_count = count_prefetch_candidates(
                    layer_id=layer_idx,
                    expert_matrix=projected,
                    future_layers=future_layers,
                    min_score=min_score,
                )
                metadata["kept_candidate_count"] = int(kept_candidate_count)
                if kept_candidate_count == 0:
                    metadata["candidate_source"] = "empty"
                    metadata["fallback_reason"] = (
                        "all_scores_below_threshold"
                        if raw_positive_count == 0
                        else "backbone_projection_empty"
                    )
                return projected, metadata

            masked_matrix = np.where(consensus_mask, base_matrix, 0.0).astype(
                np.float32,
                copy=False,
            )
            runtime_retained_count = count_prefetch_candidates(
                layer_id=layer_idx,
                expert_matrix=masked_matrix,
                future_layers=future_layers,
                min_score=min_score,
            )
            metadata["consensus_retained_count"] = int(runtime_retained_count)
            if runtime_retained_count == 0:
                metadata["consensus_fallback_used"] = True
                metadata["consensus_fallback_reason"] = "consensus_empty_fallback"
                metadata["backbone_fallback_used"] = True
                metadata["fallback_reason"] = "consensus_empty_fallback"
                metadata["candidate_source"] = "backbone_raw_projection"
                projected = self._project_backbone(base_matrix)
                kept_candidate_count = count_prefetch_candidates(
                    layer_id=layer_idx,
                    expert_matrix=projected,
                    future_layers=future_layers,
                    min_score=min_score,
                )
                metadata["kept_candidate_count"] = int(kept_candidate_count)
                if kept_candidate_count == 0:
                    metadata["candidate_source"] = "empty"
                    metadata["fallback_reason"] = (
                        "all_scores_below_threshold"
                        if raw_positive_count == 0
                        else "backbone_projection_empty"
                    )
                return projected, metadata

            projected = self._project_backbone(masked_matrix)
            kept_candidate_count = count_prefetch_candidates(
                layer_id=layer_idx,
                expert_matrix=projected,
                future_layers=future_layers,
                min_score=min_score,
            )
            metadata["kept_candidate_count"] = int(kept_candidate_count)
            if kept_candidate_count == 0:
                metadata["candidate_source"] = "empty"
                metadata["fallback_reason"] = (
                    "all_scores_below_threshold"
                    if raw_positive_count == 0
                    else "backbone_projection_empty"
                )
            return projected, metadata

        if configured_mode != "history_score" or self.policy_name != "finegrained_history_reuse":
            projected = self._project_backbone(base_matrix)
            kept_candidate_count = count_prefetch_candidates(
                layer_id=layer_idx,
                expert_matrix=projected,
                future_layers=future_layers,
                min_score=min_score,
            )
            metadata["kept_candidate_count"] = int(kept_candidate_count)
            if kept_candidate_count == 0:
                metadata["candidate_source"] = "empty"
                metadata["fallback_reason"] = (
                    "all_scores_below_threshold"
                    if raw_positive_count == 0
                    else "backbone_projection_empty"
                )
            return projected, metadata

        supports = list(
            self.policy.backbone_supports(
                seq_id=seq_id,
                layer_idx=layer_idx,
                base_matrix=base_matrix,
            )
        )
        metadata["backbone_support_count"] = len(supports)
        history_match_count = sum(
            1 for support in supports if str(support.source).startswith("history_match_")
        )
        metadata["backbone_history_match_count"] = history_match_count

        min_matches = max(
            int(getattr(self.config, "prefetch_backbone_min_matches", 0)),
            0,
        )
        if history_match_count < min_matches:
            metadata["backbone_fallback_used"] = True
            metadata["fallback_reason"] = "history_match_insufficient"
            metadata["candidate_source"] = "backbone_raw_projection"
            projected = self._project_backbone(base_matrix)
            kept_candidate_count = count_prefetch_candidates(
                layer_id=layer_idx,
                expert_matrix=projected,
                future_layers=future_layers,
                min_score=min_score,
            )
            metadata["kept_candidate_count"] = int(kept_candidate_count)
            if kept_candidate_count == 0:
                metadata["candidate_source"] = "empty"
                metadata["fallback_reason"] = (
                    "all_scores_below_threshold"
                    if raw_positive_count == 0
                    else "backbone_projection_empty"
                )
            return projected, metadata

        score_matrix = self._compute_history_backbone_score_matrix(supports)
        projected = self._project_backbone_with_scores(
            base_matrix=base_matrix,
            score_matrix=score_matrix,
        )
        kept_candidate_count = count_prefetch_candidates(
            layer_id=layer_idx,
            expert_matrix=projected,
            future_layers=future_layers,
            min_score=min_score,
        )
        metadata["kept_candidate_count"] = int(kept_candidate_count)
        if kept_candidate_count == 0:
            metadata["candidate_source"] = "empty"
            metadata["fallback_reason"] = (
                "all_scores_below_threshold"
                if raw_positive_count == 0
                else "backbone_projection_empty"
            )
        return projected, metadata

    def _record_phasea_policy_event(
        self,
        *,
        seq_id: str,
        layer_idx: int,
        expert_list,
        projected_matrix: Optional[np.ndarray],
        decision_latency_us: int,
        backbone_metadata: Optional[Dict[str, int | str | bool]] = None,
    ) -> None:
        if self.phasea_recorder is None:
            return
        if isinstance(expert_list, torch.Tensor):
            actual_expert_array = expert_list.detach().cpu().numpy()
        else:
            actual_expert_array = np.asarray(expert_list)
        self.phasea_recorder.record_policy_event(
            seq_id=seq_id,
            layer_idx=layer_idx,
            actual_expert_array=actual_expert_array,
            policy_name=self.policy_name,
            prefetch_enabled=self.prefetch_enabled(),
            score_only=bool(getattr(self.config, "policy_score_only", False)),
            decision_latency_us=decision_latency_us,
            expert_matrix=projected_matrix,
            analysis_expert_matrix=projected_matrix,
            num_layers=getattr(self.predictor, "num_layers", None),
            prefetch_future_layers=int(
                getattr(self.config, "prefetch_future_layers", 0)
            ),
            prefetch_max_candidates=int(
                getattr(self.config, "prefetch_max_candidates", 0)
            ),
            analysis_future_layers=int(
                getattr(self.config, "phasea_analysis_future_layers", 0)
            ),
            analysis_max_candidates=int(
                getattr(
                    self.config,
                    "phasea_analysis_max_ranked_candidates",
                    128,
                )
            ),
            prefetch_candidate_min_score=float(
                getattr(self.config, "prefetch_candidate_min_score", 1e-6)
            ),
            candidate_source=str(
                (backbone_metadata or {}).get("candidate_source", "empty")
            ),
            fallback_reason=str(
                (backbone_metadata or {}).get("fallback_reason", "none")
            ),
            library_match_count=int(
                (backbone_metadata or {}).get("library_match_count", 0)
            ),
            local_match_count=int(
                (backbone_metadata or {}).get("local_match_count", 0)
            ),
            raw_positive_count=int(
                (backbone_metadata or {}).get("raw_positive_count", 0)
            ),
            kept_candidate_count=int(
                (backbone_metadata or {}).get("kept_candidate_count", 0)
            ),
            all_scores_below_threshold=bool(
                (backbone_metadata or {}).get("all_scores_below_threshold", False)
            ),
            backbone_mode=str(
                (backbone_metadata or {}).get(
                    "backbone_mode",
                    getattr(self.config, "prefetch_backbone_mode", "raw_topk"),
                )
            ),
            backbone_support_count=int(
                (backbone_metadata or {}).get("backbone_support_count", 0)
            ),
            backbone_history_match_count=int(
                (backbone_metadata or {}).get("backbone_history_match_count", 0)
            ),
            backbone_projection_topk=int(
                (backbone_metadata or {}).get(
                    "backbone_projection_topk",
                    getattr(self.config, "prefetch_backbone_topk", 0),
                )
            ),
            backbone_fallback_used=bool(
                (backbone_metadata or {}).get("backbone_fallback_used", False)
            ),
            consensus_min_votes=int(
                (backbone_metadata or {}).get("consensus_min_votes", 0)
            ),
            consensus_support_count=int(
                (backbone_metadata or {}).get("consensus_support_count", 0)
            ),
            consensus_retained_count=int(
                (backbone_metadata or {}).get("consensus_retained_count", 0)
            ),
            consensus_fallback_used=bool(
                (backbone_metadata or {}).get("consensus_fallback_used", False)
            ),
            consensus_fallback_reason=str(
                (backbone_metadata or {}).get("consensus_fallback_reason", "none")
            ),
            historical_reuse_object_mode=str(
                (backbone_metadata or {}).get(
                    "historical_reuse_object_mode",
                    getattr(
                        self.config,
                        "historical_reuse_object_mode",
                        "sequence_matrix",
                    ),
                )
            ),
            local_continuation_key_layers=int(
                (backbone_metadata or {}).get(
                    "local_continuation_key_layers",
                    getattr(self.config, "local_continuation_key_layers", 4),
                )
            ),
            local_continuation_future_layers=int(
                (backbone_metadata or {}).get(
                    "local_continuation_future_layers",
                    getattr(self.config, "local_continuation_future_layers", 4),
                )
            ),
            historical_similarity_mode=str(
                getattr(self.config, "historical_library_similarity_mode", "prefix_mean")
            ),
            historical_recent_window=int(
                getattr(self.config, "historical_library_recent_window", 4)
            ),
        )

    def update_and_score(self, seq_id: str, expert_list, layer_idx: int) -> Optional[np.ndarray]:
        import time

        start_ns = time.perf_counter_ns()
        matrix = self.policy.update_and_score(seq_id, expert_list, layer_idx)
        score_metadata = self.policy.pop_score_metadata()
        projected, backbone_metadata = self._project_prefetch_matrix(
            seq_id=seq_id,
            layer_idx=layer_idx,
            base_matrix=matrix,
            score_metadata=score_metadata,
        )
        decision_latency_us = int((time.perf_counter_ns() - start_ns) / 1000)
        self._record_phasea_policy_event(
            seq_id=seq_id,
            layer_idx=layer_idx,
            expert_list=expert_list,
            projected_matrix=projected,
            decision_latency_us=decision_latency_us,
            backbone_metadata=backbone_metadata,
        )
        return projected

    def update_only(self, seq_id: str, expert_list, layer_idx: int) -> None:
        self.policy.update(seq_id, expert_list, layer_idx)
        self._record_phasea_policy_event(
            seq_id=seq_id,
            layer_idx=layer_idx,
            expert_list=expert_list,
            projected_matrix=None,
            decision_latency_us=0,
            backbone_metadata={
                "backbone_mode": "disabled",
                "backbone_support_count": 0,
                "backbone_history_match_count": 0,
                "backbone_projection_topk": int(
                    getattr(self.config, "prefetch_backbone_topk", 0)
                ),
                "backbone_fallback_used": False,
                "consensus_min_votes": int(
                    getattr(self.config, "historical_reuse_consensus_min_votes", 1)
                ),
                "consensus_support_count": 0,
                "consensus_retained_count": 0,
                "consensus_fallback_used": False,
                "consensus_fallback_reason": "none",
            },
        )

    def prefetch_enabled(self) -> bool:
        return bool(getattr(self.config, "prefetch", False))

    def finish_sequence(self, seq_id: str) -> None:
        self.policy.finish_sequence(seq_id)
        if self.phasea_recorder is not None:
            self.phasea_recorder.record_sequence_finish(seq_id)

    def library_size(self) -> int:
        object_mode = getattr(
            self.config,
            "historical_reuse_object_mode",
            "sequence_matrix",
        )
        if object_mode == "local_continuation":
            return len(self.local_library)
        return len(self.library)

    def library_stats(self) -> Dict[str, int]:
        object_mode = getattr(
            self.config,
            "historical_reuse_object_mode",
            "sequence_matrix",
        )
        if object_mode == "local_continuation":
            stats = dict(self.local_library.stats())
            stats["sequence_library_size"] = len(self.library)
            return stats
        return self.library.stats()
