# Original MoE-Infinity Workflow Audit

Date: 2026-04-30

## Summary

The upstream open-source repository on the A800 server is:

- `/data/ziheng/projects/MoE-Infinity`
- remote: `https://github.com/EfficientMoE/MoE-Infinity.git`

The upstream README explicitly says the open-source runtime is not the same as
the paper's extreme-performance version. Therefore, the paper's reported
performance should not be treated as the default behavior of the open-source
workflow.

The open-source README-style workflow is:

```python
from moe_infinity import MoE

config = {
    "offload_path": "...",
    "device_memory_ratio": 0.75,
}
model = MoE(checkpoint, config)
output_ids = model.generate(input_ids)
```

In this workflow, `ArcherConfig.prefetch` defaults to `False`, and the README
examples do not enable it.

## Prefetch Path in Upstream

When a model block actually calls the upstream prefetcher, the path is:

```text
MoE.generate()
  -> MoE block forward
  -> router top-k selected experts
  -> ExpertPredictor.predict(seq_id, selected_experts, layer_id)
  -> ExpertTracer.find_most_similar(...)
  -> ExpertPrefetcher.prefetch_experts(layer_id, expert_matrix)
  -> archer_engine.replace_cache_candidates(tensor_ids)
  -> archer_engine.enqueue_prefetch(tensor_id, gpu_id)
  -> ArcherTaskPool background priority-1 prefetch
```

This path has two distinct mechanisms:

- `replace_cache_candidates()` updates the candidate set used by eviction. This
  is a retention/protection mechanism.
- `enqueue_prefetch()` creates background transfer work. This is true H2D
  prefetch.

These mechanisms must be measured separately. A miss reduction with
`queue_push == 0` is not evidence of H2D prefetch benefit.

## Model-Specific Status in Upstream

The current upstream open-source code does not enable the same prefetch path for
all supported models.

| Model wrapper | Forward-level predictor/prefetch call |
| --- | --- |
| `moe_infinity/models/arctic.py` | enabled |
| `moe_infinity/models/grok.py` | enabled |
| `moe_infinity/models/mixtral.py` | commented out |
| `moe_infinity/models/nllb_moe.py` | commented out |
| `moe_infinity/models/switch_transformers.py` | commented out |
| `moe_infinity/models/deepseek.py` | absent |
| `moe_infinity/models/qwen.py` | absent |

This means Qwen/DeepSeek experiments in `moe_infinity_fgo` are not measuring an
unchanged upstream activation-aware prefetch path. The fgo branch adds its own
`drive_expert_policy()` hook to Qwen/DeepSeek/Mixtral-style wrappers.

## Required Parity Check

Before claiming a runtime improvement over MoE-Infinity, run
`benchmarks/original_workflow_parity_v1.py` and report:

- whether upstream can instantiate the target model;
- whether upstream calls `ExpertPrefetcher.prefetch_experts()`;
- whether fgo on-demand matches the same offloading/cache baseline;
- whether fgo prefetch effects come from retention or true H2D transfer.

If upstream does not support the target model or does not call the prefetcher,
the result is a workflow finding, not a performance comparison.
