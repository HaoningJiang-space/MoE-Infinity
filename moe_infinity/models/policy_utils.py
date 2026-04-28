from __future__ import annotations


def drive_expert_policy(module, expert_index) -> None:
    expert_policy = getattr(module, "expert_policy", None)
    expert_prefetcher = getattr(module, "expert_prefetcher", None)
    seq_id_list = getattr(module, "seq_id_list", None)
    if expert_policy is None or expert_prefetcher is None or seq_id_list is None:
        return

    batch_size = expert_index.shape[0]
    enable_prefetch = getattr(module, "enable_expert_prefetch", True)
    score_only = getattr(module, "expert_policy_score_only", False)
    for batch_idx in range(batch_size):
        seq_id = seq_id_list[batch_idx]
        if not enable_prefetch:
            if score_only:
                expert_policy.update_and_score(
                    seq_id,
                    expert_index[batch_idx],
                    module.layer_id,
                )
            else:
                expert_policy.update_only(
                    seq_id,
                    expert_index[batch_idx],
                    module.layer_id,
                )
            continue

        credit = None
        credit_enabled = bool(
            getattr(expert_prefetcher, "prefetch_credit_gated_enabled", False)
        )
        if credit_enabled and hasattr(expert_prefetcher, "speculation_credit"):
            credit = int(expert_prefetcher.speculation_credit())
            if credit <= 0:
                expert_policy.update_only(
                    seq_id,
                    expert_index[batch_idx],
                    module.layer_id,
                )
                if hasattr(expert_prefetcher, "record_credit_skip"):
                    expert_prefetcher.record_credit_skip()
                continue

        expert_matrix = expert_policy.update_and_score(
            seq_id,
            expert_index[batch_idx],
            module.layer_id,
        )
        if expert_matrix is None:
            continue
        expert_prefetcher.prefetch_experts(
            module.layer_id,
            expert_matrix,
            max_candidates_override=credit,
        )
