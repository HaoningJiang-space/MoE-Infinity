from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .history_reuse import FineGrainedHistoryReusePolicy
from .historical_library import HistoricalExpertLibrary
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
            admission_policy=config.historical_library_admission,
            dedup_threshold=config.historical_library_dedup_threshold,
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
                model_tag=model_tag,
            ),
            "static_frequency": StaticFrequencyPolicy(
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

    @property
    def policy(self):
        return self.policies[self.policy_name]

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

    def update_and_score(self, seq_id: str, expert_list, layer_idx: int) -> Optional[np.ndarray]:
        matrix = self.policy.update_and_score(seq_id, expert_list, layer_idx)
        return self._project_backbone(matrix)

    def update_only(self, seq_id: str, expert_list, layer_idx: int) -> None:
        self.policy.update(seq_id, expert_list, layer_idx)

    def prefetch_enabled(self) -> bool:
        return bool(getattr(self.config, "prefetch", False))

    def finish_sequence(self, seq_id: str) -> None:
        self.policy.finish_sequence(seq_id)

    def library_size(self) -> int:
        return len(self.library)

    def library_stats(self) -> Dict[str, int]:
        return self.library.stats()
