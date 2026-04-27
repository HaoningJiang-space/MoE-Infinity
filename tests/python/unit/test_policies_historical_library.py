from types import SimpleNamespace

import numpy as np
import pytest

from moe_infinity.memory.expert_tracer import ExpertTracer
from moe_infinity.policies.historical_library import HistoricalExpertLibrary


def _fake_cfg(num_layers=3, num_experts=4):
    return SimpleNamespace(
        architectures=["Qwen3MoeForCausalLM"],
        num_hidden_layers=num_layers,
        num_experts=num_experts,
    )


@pytest.fixture(autouse=True)
def reset_tracer_singleton():
    ExpertTracer._instance = None
    yield
    ExpertTracer._instance = None


def test_historical_library_topk_prefers_prefix_match():
    library = HistoricalExpertLibrary(capacity=4, metric="cosine")
    a = np.array(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float32,
    )
    b = np.array(
        [[0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    library.admit("a", a, model_tag="qwen")
    library.admit("b", b, model_tag="qwen")

    current = np.array(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    matches = library.topk(current, layer_idx=1, model_tag="qwen", k=2)

    assert [entry.seq_id for entry, _ in matches] == ["a", "b"]
    assert matches[0][1] > matches[1][1]


def test_historical_library_deduplicates_near_identical_entries():
    library = HistoricalExpertLibrary(
        capacity=2,
        metric="cosine",
        dedup_threshold=0.99,
    )
    base = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    library.admit("first", base, model_tag="mixtral")
    library.admit("second", base * 2.0, model_tag="mixtral")

    assert len(library) == 1
    assert library.entries[0].access_count == 2
    assert library.entries[0].seq_id == "first"


def test_historical_library_respects_capacity_and_frequency_weighted_eviction():
    library = HistoricalExpertLibrary(
        capacity=2,
        metric="cosine",
        admission_policy="frequency_weighted",
    )
    e0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    e1 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    e2 = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    library.admit("e0", e0, model_tag="deepseek")
    library.admit("e1", e1, model_tag="deepseek")
    library.entries[0].access_count = 5
    library.entries[1].access_count = 1
    library.admit("e2", e2, model_tag="deepseek")

    seq_ids = {entry.seq_id for entry in library.entries}
    assert seq_ids == {"e0", "e2"}

