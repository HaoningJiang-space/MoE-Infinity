from __future__ import annotations

import numpy as np

from .base import OffloadingPolicy
from .historical_library import HistoricalExpertLibrary


class FineGrainedHistoryReusePolicy(OffloadingPolicy):
    def __init__(self, *, config, tracer, predictor, library: HistoricalExpertLibrary, model_tag: str = ""):
        super().__init__(config=config, tracer=tracer, predictor=predictor, model_tag=model_tag)
        self.library = library

    def score_for_prefetch(self, seq_id: str, layer_idx: int) -> np.ndarray:
        current_entry = self.tracer.get_entry(seq_id)
        matches = self.library.topk(
            current_entry.matrix,
            layer_idx=layer_idx,
            model_tag=self.model_tag,
            k=1,
        )
        if not matches:
            return self.predictor.predict_from_current_trace(seq_id, layer_idx)
        matched_entry, _ = matches[0]
        matched_entry.access_count += 1
        return self.predictor.build_prefetch_matrix(
            matched_entry.matrix,
            layer_idx,
        )

    def finish_sequence(self, seq_id: str) -> None:
        entry = self.tracer.get_entry(seq_id)
        self.library.admit(
            seq_id,
            entry.matrix,
            model_tag=self.model_tag,
            metadata={"num_new_tokens": entry.num_new_tokens},
        )
