from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from moe_infinity.memory.expert_prefetcher import ExpertPrefetcher


def _fake_model_cfg():
    return SimpleNamespace(
        architectures=["Qwen2MoeForCausalLM"],
        num_hidden_layers=6,
        num_experts=5,
    )


def test_prefetch_experts_respects_future_layer_and_candidate_limits():
    prefetcher = ExpertPrefetcher(_fake_model_cfg())
    prefetcher.archer_engine = Mock()
    prefetcher.archer_engine.get_node_default_device.return_value = 0
    prefetcher.prefetch_future_layers = 2
    prefetcher.prefetch_max_candidates = 3
    prefetcher.expert_tensor_map = {
        (layer, expert): layer * 10 + expert
        for layer in range(prefetcher.num_layers)
        for expert in range(prefetcher.num_experts)
    }
    matrix = np.zeros(
        (prefetcher.num_layers, prefetcher.num_experts), dtype=np.float32
    )
    matrix[1, 0] = 0.1
    matrix[2, 3] = 0.9
    matrix[2, 4] = 0.8
    matrix[3, 1] = 0.7
    matrix[4, 2] = 0.95  # outside the future-layer budget

    prefetcher.prefetch_experts(1, matrix)

    prefetcher.archer_engine.replace_cache_candidates.assert_called_once_with(
        [23, 24, 31]
    )
    enqueued = [
        call.args[0]
        for call in prefetcher.archer_engine.enqueue_prefetch.call_args_list
    ]
    assert enqueued == [23, 24, 31]
