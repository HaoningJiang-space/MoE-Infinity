from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from moe_infinity.models.policy_utils import drive_expert_policy


def test_drive_expert_policy_prefetches_per_batch_item():
    policy = Mock()
    policy.update_and_score.side_effect = [
        np.ones((2, 3), dtype=np.float32),
        None,
    ]
    prefetcher = Mock()
    module = SimpleNamespace(
        expert_policy=policy,
        expert_prefetcher=prefetcher,
        seq_id_list=["seq0", "seq1"],
        layer_id=4,
    )
    expert_index = np.array(
        [
            [[0, 1], [1, 2]],
            [[2, 1], [0, 1]],
        ]
    )

    drive_expert_policy(module, expert_index)

    assert policy.update_and_score.call_count == 2
    first_call = policy.update_and_score.call_args_list[0]
    second_call = policy.update_and_score.call_args_list[1]
    assert first_call.args[0] == "seq0"
    assert second_call.args[0] == "seq1"
    assert prefetcher.prefetch_experts.call_count == 1
    assert prefetcher.prefetch_experts.call_args.args[0] == 4


def test_drive_expert_policy_can_skip_policy_drive():
    policy = Mock()
    prefetcher = Mock()
    prefetcher.prefetch_policy_disabled = True
    module = SimpleNamespace(
        expert_policy=policy,
        expert_prefetcher=prefetcher,
        seq_id_list=["seq0"],
        layer_id=4,
    )
    expert_index = np.array([[[0, 1], [1, 2]]])

    drive_expert_policy(module, expert_index)

    policy.update_and_score.assert_not_called()
    policy.update_only.assert_not_called()
    prefetcher.prefetch_experts.assert_not_called()
