from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExpertTraceEntry:
    seq_id: str = None
    matrix: np.ndarray = None
    access: int = 0
    num_new_tokens: int = 0
    current_step_index: int = 0
    step_matrices: list[np.ndarray] = field(default_factory=list)

    def __hash__(self):
        return hash(self.seq_id)


@dataclass
class ExpertCacheEntry:
    expert_idx: int = None
    layer_idx: int = None
    r: float = 0.0
    visit: int = 0
    timestamp: int = 0

    def __hash__(self):
        return hash((self.layer_idx, self.expert_idx))
