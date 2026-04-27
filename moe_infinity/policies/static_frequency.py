from __future__ import annotations

import numpy as np

from .base import OffloadingPolicy


class StaticFrequencyPolicy(OffloadingPolicy):
    def score_for_prefetch(self, seq_id: str, layer_idx: int) -> np.ndarray:
        aggregate = np.zeros((self.predictor.num_layers, self.predictor.num_experts), dtype=np.float32)
        trace_collection = self.tracer.trace_collection.detach().cpu().numpy()
        if trace_collection.shape[0] > 0:
            aggregate += np.sum(trace_collection, axis=0)
        for entry in self.tracer.trace.values():
            if entry.seq_id == seq_id:
                continue
            aggregate += entry.matrix
        return self.predictor.build_prefetch_matrix(aggregate, layer_idx)
