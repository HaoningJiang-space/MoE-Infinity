from types import SimpleNamespace

import numpy as np
import pytest

from moe_infinity.memory.expert_predictor import ExpertPredictor
from moe_infinity.memory.expert_tracer import ExpertTracer
from moe_infinity.policies import OffloadingPolicyManager


def _fake_model_cfg(num_layers=3, num_experts=4):
    return SimpleNamespace(
        architectures=["Qwen3MoeForCausalLM"],
        num_hidden_layers=num_layers,
        num_experts=num_experts,
    )


def _fake_archer_cfg(**overrides):
    base = {
        "offloading_policy": "baseline_trace_similarity",
        "historical_library_capacity": 8,
        "historical_library_metric": "cosine",
        "historical_library_admission": "diversity_aware",
        "historical_library_dedup_threshold": 0.995,
        "prefetch_backbone_topk": 0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def reset_tracer_singleton():
    ExpertTracer._instance = None
    yield
    ExpertTracer._instance = None


def _make_manager(*, policy="baseline_trace_similarity", topk=0):
    model_cfg = _fake_model_cfg()
    tracer = ExpertTracer(capacity=8, config=model_cfg)
    predictor = ExpertPredictor(model_cfg)
    predictor.add_tracer(tracer)
    manager = OffloadingPolicyManager(
        config=_fake_archer_cfg(
            offloading_policy=policy,
            prefetch_backbone_topk=topk,
        ),
        tracer=tracer,
        predictor=predictor,
        model_tag="qwen",
    )
    return manager, tracer


def test_policy_manager_backbone_projection_keeps_topk_per_layer():
    manager, _ = _make_manager(topk=2)
    matrix = np.array(
        [
            [0.0, 0.1, 0.4, 0.3],
            [0.2, 0.0, 0.6, 0.5],
            [0.7, 0.1, 0.2, 0.0],
        ],
        dtype=np.float32,
    )

    projected = manager._project_backbone(matrix)

    assert np.count_nonzero(projected[0]) == 2
    assert np.count_nonzero(projected[1]) == 2
    assert np.count_nonzero(projected[2]) == 2
    assert projected[0, 2] > 0 and projected[0, 3] > 0
    assert projected[1, 2] > 0 and projected[1, 3] > 0
    assert projected[2, 0] > 0 and projected[2, 2] > 0


def test_history_reuse_policy_admits_finished_sequence():
    manager, tracer = _make_manager(policy="finegrained_history_reuse")
    seq_id = tracer.create_entry()
    tracer.update_entry(seq_id, np.array([[0, 1], [1, 2]]), layer_idx=0)
    tracer.update_entry(seq_id, np.array([[2, 3], [3, 0]]), layer_idx=1)

    manager.finish_sequence(seq_id)

    assert manager.library_size() == 1
    entry = manager.library.entries[0]
    assert entry.seq_id == seq_id
    assert entry.model_tag == "qwen"


def test_static_frequency_policy_uses_existing_history():
    manager, tracer = _make_manager(policy="static_frequency")
    seq_a = tracer.create_entry()
    seq_b = tracer.create_entry()
    tracer.update_entry(seq_a, np.array([[0, 0], [0, 0]]), layer_idx=0)
    tracer.update_entry(seq_b, np.array([[2, 2], [2, 2]]), layer_idx=0)
    tracer.update_entry(seq_b, np.array([[1, 1], [1, 1]]), layer_idx=1)

    matrix = manager.update_and_score(
        seq_a,
        np.array([[0, 1], [1, 0]]),
        layer_idx=1,
    )

    assert matrix is not None
    assert matrix.shape == (3, 4)
    assert np.count_nonzero(matrix) > 0


def test_static_hot_policy_does_not_update_trace():
    manager, tracer = _make_manager(policy="static_hot_prefetch")
    seq_id = tracer.create_entry()

    matrix = manager.update_and_score(
        seq_id,
        np.array([[2, 3], [3, 0]]),
        layer_idx=1,
    )

    assert matrix is not None
    assert matrix.shape == (3, 4)
    assert np.count_nonzero(matrix) > 0
    assert np.count_nonzero(tracer.trace[seq_id].matrix) == 0
