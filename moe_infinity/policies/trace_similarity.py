from __future__ import annotations

import numpy as np

from .base import OffloadingPolicy


class TraceSimilarityPolicy(OffloadingPolicy):
    def score_for_prefetch(self, seq_id: str, layer_idx: int) -> np.ndarray:
        return self.predictor.predict_from_current_trace(seq_id, layer_idx)
