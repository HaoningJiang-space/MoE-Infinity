from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class OffloadingPolicy(ABC):
    def __init__(self, *, config, tracer, predictor, model_tag: str = ""):
        self.config = config
        self.tracer = tracer
        self.predictor = predictor
        self.model_tag = model_tag

    def update(self, seq_id: str, expert_list, layer_idx: int) -> None:
        self.tracer.update_entry(seq_id, expert_list, layer_idx)

    @abstractmethod
    def score_for_prefetch(self, seq_id: str, layer_idx: int) -> Optional[np.ndarray]:
        raise NotImplementedError

    def score_for_retention(self, seq_id: str, layer_idx: int) -> Optional[np.ndarray]:
        return None

    def finish_sequence(self, seq_id: str) -> None:
        return None

    def update_and_score(self, seq_id: str, expert_list, layer_idx: int) -> Optional[np.ndarray]:
        self.update(seq_id, expert_list, layer_idx)
        return self.score_for_prefetch(seq_id, layer_idx)
