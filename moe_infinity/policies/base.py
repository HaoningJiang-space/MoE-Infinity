from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class BackboneSupport:
    matrix: np.ndarray
    weight: float = 1.0
    source: str = "base"


class OffloadingPolicy(ABC):
    def __init__(self, *, config, tracer, predictor, model_tag: str = ""):
        self.config = config
        self.tracer = tracer
        self.predictor = predictor
        self.model_tag = model_tag
        self._last_score_metadata = {}

    def update(self, seq_id: str, expert_list, layer_idx: int) -> None:
        self.tracer.update_entry(seq_id, expert_list, layer_idx)

    @abstractmethod
    def score_for_prefetch(self, seq_id: str, layer_idx: int) -> Optional[np.ndarray]:
        raise NotImplementedError

    def score_for_retention(self, seq_id: str, layer_idx: int) -> Optional[np.ndarray]:
        return None

    def backbone_supports(
        self,
        seq_id: str,
        layer_idx: int,
        base_matrix: np.ndarray,
    ) -> List[BackboneSupport]:
        return [
            BackboneSupport(
                matrix=np.asarray(base_matrix, dtype=np.float32).copy(),
                weight=1.0,
                source="base",
            )
        ]

    def finish_sequence(self, seq_id: str) -> None:
        return None

    def clear_score_metadata(self) -> None:
        self._last_score_metadata = {}

    def set_score_metadata(self, **metadata) -> None:
        self._last_score_metadata = dict(metadata)

    def pop_score_metadata(self) -> dict:
        metadata = dict(self._last_score_metadata)
        self._last_score_metadata = {}
        return metadata

    def update_and_score(self, seq_id: str, expert_list, layer_idx: int) -> Optional[np.ndarray]:
        self.clear_score_metadata()
        self.update(seq_id, expert_list, layer_idx)
        return self.score_for_prefetch(seq_id, layer_idx)
