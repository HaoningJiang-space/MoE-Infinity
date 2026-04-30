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

## Server Smoke Results

### 1. Same-env import check

The first upstream-only check was run at:

- `/data/ziheng/moe_infinity_fgo_runs/original_workflow_parity_v1_qwen_importcheck`

Both `upstream_readme_default` and `upstream_prefetch_flag` fail before model
construction in the current `mxmoe` environment:

```text
ImportError: cannot import name 'is_torch_fx_available'
from 'transformers.utils.import_utils'
```

This means the upstream open-source workflow is not currently runnable in the
same Python environment used by `moe_infinity_fgo`. A fair upstream comparison
requires either a compatible upstream environment or an explicit statement that
the comparison is against the fgo open-source-derived runtime, not the upstream
README workflow.

### 2. Compatible-env upstream workflow check

A second upstream-only smoke was run in an isolated environment:

- env: `/data/ziheng/conda_envs/moeinf-upstream`
- torch: `2.9.1+cu128`
- transformers: `4.53.0`
- upstream repo: `/data/ziheng/projects/MoE-Infinity`
- result root:
  `/data/ziheng/moe_infinity_fgo_runs/original_workflow_parity_v1_upstream_env_check`

The upstream extension required one compile-compatibility patch in
`core/parallel/expert_dispatcher.cpp`: several `kNumDevices` uses had to be
changed to `kNumDevices()`. This does not change prefetch policy, but it must be
reported as an environment compatibility patch.

Because upstream DeepSeek cannot call `generate()` with the current
Transformers interface, the probe falls back to one `forward()` pass. Therefore
this smoke is workflow evidence only, not a latency comparison.

| Case | Execution mode | Success | Prefetch API calls |
| --- | --- | ---: | ---: |
| `upstream_readme_default` | `forward_fallback` | yes | `0` |
| `upstream_prefetch_flag` | `forward_fallback` | yes | `0` |

This confirms the source-code audit for DeepSeek: setting `prefetch=true` in the
open-source config does not by itself make the DeepSeek wrapper call
`ExpertPrefetcher.prefetch_experts()`.

### 3. Source-only generate baseline recovery

The valid recovery path is not a custom decode loop. The source-only recovery
uses:

- upstream worktree:
  `/data/ziheng/projects/MoE-Infinity-generate-compat`
- upstream commit: `d617801`
- environment:
  `/data/ziheng/conda_envs/moeinf-upstream-generate`
- `transformers`: `4.40.2`
- benchmark driver:
  `benchmarks/original_upstream_baseline_v1.py`
- source workflow: upstream `MoE.generate()`

Smoke result:

- root:
  `/data/ziheng/moe_infinity_fgo_runs/original_upstream_source_baseline_tf440_smoke`
- cold source-only baseline: `12.39 tok/s`, `80.69 ms/token`
- warm source-only baseline: `12.78 tok/s`, `78.26 ms/token`
- `ExpertPrefetcher.prefetch_experts()` calls: `0`

This is the first valid source-only DeepSeek baseline in the current server
setup. It also confirms that the upstream DeepSeek path is an offloading
baseline, not an activation-aware prefetch baseline.
