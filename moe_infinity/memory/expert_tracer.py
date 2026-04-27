import copy
import os
import uuid
from collections import Counter
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from transformers import PretrainedConfig

# from sklearn.metrics.pairwise import cosine_similarity
from moe_infinity.memory.expert_entry import ExpertTraceEntry
from moe_infinity.utils import parse_moe_param


class ExpertTracer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExpertTracer, cls).__new__(cls)
        return cls._instance

    def __init__(self, capacity: int, config: PretrainedConfig):
        self.num_layers, self.num_experts, self.num_encoder_layers = (
            parse_moe_param(config)
        )
        self.capacity = capacity
        self.trace_device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.trace = {}

        self.trace_collection = torch.zeros(
            (capacity, self.num_layers, self.num_experts),
            device=self.trace_device,
            dtype=torch.float32,
        )
        self.persistent_capacity = 0
        self.collection_access = np.zeros((capacity,))

        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def load_trace(self, trace: Union[os.PathLike, np.ndarray]):
        if isinstance(trace, os.PathLike):
            trace = np.load(trace, allow_pickle=False)

        if isinstance(trace, np.ndarray):
            self.trace_collection = torch.as_tensor(
                trace, dtype=torch.float32, device=self.trace_device
            )
        elif isinstance(trace, torch.Tensor):
            self.trace_collection = torch.as_tensor(
                trace, dtype=torch.float32, device=self.trace_device
            )
        else:
            raise TypeError(f"Unsupported trace type: {type(trace)!r}")

        self.persistent_capacity = self.trace_collection.shape[0]
        assert self.persistent_capacity <= self.capacity, (
            f"loaded trace capacity {self.persistent_capacity} must be "
            f"less than or equal to capacity in config {self.capacity}"
        )

    def create_entry(self):
        seq_id = uuid.uuid4().hex
        zero_matrix = np.zeros((self.num_layers, self.num_experts), dtype=np.float32)
        self.trace[seq_id] = ExpertTraceEntry(
            seq_id=seq_id,
            matrix=zero_matrix.copy(),
            access=0,
            num_new_tokens=0,
            current_step_index=0,
            step_matrices=[zero_matrix.copy()],
        )
        return seq_id

    def finish_entry(self, seq_id):
        trace_sum = torch.sum(self.trace_collection, dim=(1, 2)).cpu().numpy()

        if np.any(trace_sum == 0):
            # find the first zero entry
            idx = np.argwhere(trace_sum == 0)[0][0]
            self.trace_collection[idx] = torch.as_tensor(
                self.trace[seq_id].matrix,
                dtype=self.trace_collection.dtype,
                device=self.trace_device,
            )
            self.collection_access[idx] = 1
        else:
            # find the first entry after self.persistent_capacity that has the least access
            collection_access_copy = self.collection_access.copy()
            collection_access_copy[: self.persistent_capacity] = 1e9

            idx = np.argmin(collection_access_copy)
            self.trace_collection[idx] = torch.as_tensor(
                self.trace[seq_id].matrix,
                dtype=self.trace_collection.dtype,
                device=self.trace_device,
            )
            self.collection_access[idx] = 1

    def remove_entry(self, seq_id):
        self.trace.pop(seq_id, None)

    def update_entry(self, seq_id, expert_list, layer_idx):
        expert_counter = Counter(expert_list.flatten().tolist())
        entry = self.trace[seq_id]
        for key, count in expert_counter.items():
            entry.matrix[layer_idx, key] += count
            entry.step_matrices[entry.current_step_index][layer_idx, key] += count

        if layer_idx == self.num_layers - 1:
            entry.num_new_tokens += 1
            entry.current_step_index += 1
            entry.step_matrices.append(
                np.zeros((self.num_layers, self.num_experts), dtype=np.float32)
            )

    def get_entry_decoder(self, seq_id):
        entry = copy.deepcopy(self.trace[seq_id])
        entry.matrix[: self.num_encoder_layers, :] = 0
        return entry

    def get_entry(self, seq_id):
        return self.trace[seq_id]

    def get_current_step_matrix(self, seq_id):
        entry = self.trace[seq_id]
        return entry.step_matrices[entry.current_step_index]

    def get_completed_step_matrices(self, seq_id):
        entry = self.trace[seq_id]
        if entry.num_new_tokens <= 0:
            return []
        return [
            matrix.copy()
            for matrix in entry.step_matrices[: entry.num_new_tokens]
        ]

    def find_most_similar(self, matrix, layer_idx) -> np.ndarray:
        # start_time = time.time()
        trace_collection_copy = self.trace_collection.clone()
        trace_collection_copy[:, : (layer_idx + 1), :] = 1e-9
        # print("trace_collection copy", time.time() - start_time)

        trace_collection_copy /= torch.sum(
            trace_collection_copy, dim=2, keepdims=True
        )

        matrix_copy = torch.from_numpy(matrix.copy()).to(self.trace_device)
        matrix_copy /= torch.sum(matrix_copy, dim=1, keepdims=True)
        replicated_matrix_copy = torch.concat(
            [matrix_copy[None, ...]] * self.capacity, dim=0
        )

        # fill nan with 0 using torch
        replicated_matrix_copy = torch.nan_to_num(replicated_matrix_copy)
        matrix_copy = torch.nan_to_num(matrix_copy)

        cos_sim = self.cos(replicated_matrix_copy, trace_collection_copy)
        # print("cos_sim", time.time() - start_time)

        # print(cos_sim.shape)

        cos_dist = 1 - torch.mean(cos_sim, dim=1)
        min_idx = torch.argmin(cos_dist).item()

        self.collection_access[min_idx] += 1

        entry = self.trace_collection[min_idx].to("cpu").numpy()
        return entry
