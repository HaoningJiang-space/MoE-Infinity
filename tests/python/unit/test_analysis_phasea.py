import json

from moe_infinity.analysis.phasea import analyze_phasea_event_file


def _policy_event(seq_id: str, *, is_warmup: bool) -> dict:
    return {
        "event_type": "policy_decision",
        "seq_id": seq_id,
        "request_id": seq_id,
        "trace_name": "unit",
        "tag": "test",
        "variant": "history_reuse_local_backbone",
        "is_warmup": is_warmup,
        "step_index": 0,
        "layer_idx": 0,
        "actual_expert_counts": {"1": 1},
        "actual_experts": [1],
        "runtime_candidate_pairs": [],
        "analysis_candidate_pairs": [],
        "candidate_pairs": [],
        "decision_latency_us": 10,
        "prefetch_future_layers": 4,
        "runtime_prefetch_future_layers": 4,
    }


def test_analyze_phasea_event_file_can_exclude_warmup_events(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [
        _policy_event("warmup-seq", is_warmup=True),
        _policy_event("measured-seq", is_warmup=False),
    ]
    path.write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )

    included = analyze_phasea_event_file(path)
    excluded = analyze_phasea_event_file(path, exclude_warmup_events=True)

    assert included["instrumentation_validation"]["policy_event_count"] == 2
    assert included["instrumentation_validation"]["warmup_policy_event_count"] == 1
    assert included["instrumentation_validation"]["excluded_warmup_event_count"] == 0
    assert excluded["instrumentation_validation"]["policy_event_count"] == 1
    assert excluded["instrumentation_validation"]["warmup_policy_event_count"] == 1
    assert excluded["instrumentation_validation"]["excluded_warmup_event_count"] == 1
