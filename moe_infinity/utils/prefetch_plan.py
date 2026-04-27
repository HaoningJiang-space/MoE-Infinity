from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class PrefetchCandidate:
    layer_idx: int
    expert_idx: int
    score: float


def rank_prefetch_candidates(
    *,
    layer_id: int,
    expert_matrix: np.ndarray,
    future_layers: int = 0,
    max_candidates: int = 0,
    min_score: float = 1e-6,
) -> List[PrefetchCandidate]:
    if expert_matrix.ndim != 2:
        raise ValueError(
            f"Expected expert_matrix to be 2D, got shape {expert_matrix.shape!r}"
        )
    num_layers, num_experts = expert_matrix.shape
    max_layer = num_layers
    if future_layers > 0:
        max_layer = min(num_layers, layer_id + 1 + future_layers)

    ranked: List[PrefetchCandidate] = []
    for future_layer_idx in range(layer_id + 1, max_layer):
        row = expert_matrix[future_layer_idx]
        for expert_idx in range(num_experts):
            score = float(row[expert_idx])
            if score <= float(min_score):
                continue
            ranked.append(
                PrefetchCandidate(
                    layer_idx=future_layer_idx,
                    expert_idx=expert_idx,
                    score=score,
                )
            )

    ranked.sort(key=lambda candidate: candidate.score, reverse=True)
    if max_candidates > 0:
        ranked = ranked[:max_candidates]
    return ranked


def count_prefetch_candidates(
    *,
    layer_id: int,
    expert_matrix: np.ndarray,
    future_layers: int = 0,
    min_score: float = 1e-6,
) -> int:
    if expert_matrix.ndim != 2:
        raise ValueError(
            f"Expected expert_matrix to be 2D, got shape {expert_matrix.shape!r}"
        )
    num_layers, num_experts = expert_matrix.shape
    max_layer = num_layers
    if future_layers > 0:
        max_layer = min(num_layers, layer_id + 1 + future_layers)

    count = 0
    for future_layer_idx in range(layer_id + 1, max_layer):
        row = expert_matrix[future_layer_idx]
        for expert_idx in range(num_experts):
            if float(row[expert_idx]) > float(min_score):
                count += 1
    return int(count)


def unique_expert_indices(expert_indices: Sequence[int] | np.ndarray) -> List[int]:
    array = np.asarray(expert_indices, dtype=np.int64).reshape(-1)
    if array.size == 0:
        return []
    return [int(x) for x in np.unique(array)]


def unique_candidate_experts(
    candidates: Iterable[PrefetchCandidate],
) -> List[int]:
    experts = sorted({int(candidate.expert_idx) for candidate in candidates})
    return experts
