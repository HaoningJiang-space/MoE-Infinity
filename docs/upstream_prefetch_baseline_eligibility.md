# Upstream Prefetch Baseline Eligibility

Date: 2026-05-01

## Principle

For MoE-Infinity comparisons, a valid upstream baseline must use the original
source workflow:

```python
from moe_infinity import MoE

model = MoE(checkpoint, config)
output_ids = model.generate(input_ids)
```

Custom decode loops, manual KV-cache propagation, and manual prefetch calls are
diagnostic only. They must not be reported as original MoE-Infinity baseline
performance.

## Source-Code Finding

The open-source upstream prefetch path is model-wrapper dependent. In the
current upstream-compatible source tree:

| Wrapper | Forward-level `ExpertPrefetcher.prefetch_experts()` call |
| --- | --- |
| `moe_infinity/models/arctic.py` | enabled |
| `moe_infinity/models/grok.py` | enabled |
| `moe_infinity/models/mixtral.py` | commented out |
| `moe_infinity/models/nllb_moe.py` | commented out |
| `moe_infinity/models/switch_transformers.py` | commented out |
| `moe_infinity/models/deepseek.py` | commented out / absent from active forward path |
| `moe_infinity/models/qwen.py` | absent in upstream source-only path |

Therefore, setting `prefetch=true` is not sufficient. The model wrapper must
actually call the prefetcher in its forward path.

## Server Evidence

Formal DeepSeek source-only run:

- Root: `/data/ziheng/moe_infinity_fgo_runs/source_only_fair_matrix_v2_deepseek_readme_per64_r1`
- Model: `/data/ziheng/models/DeepSeek-V2-Lite`
- Workload: README mixed mirror, 64 measured requests, 16 generated tokens
- Workflow: upstream `MoE(...).generate`

| Case | Label | TPOT | tok/s | Prefetch calls |
| --- | --- | ---: | ---: | ---: |
| `upstream_plain` | source-only baseline | `113.049 ms` | `8.846` | `0` |
| `upstream_counter` | source-only baseline | `107.920 ms` | `9.266` | `0` |
| `fgo_plain` | environment failure | N/A | N/A | `0` |
| `fgo_counter` | environment failure | N/A | N/A | `0` |

FGO DeepSeek compatibility smoke:

- Root: `/data/ziheng/moe_infinity_fgo_runs/source_only_fair_matrix_v2_deepseek_fgo_compat453_smoke`
- `transformers=4.53.0`
- Result: `DeepseekV2ForCausalLM` has no `generate` method in this source path.

Available-model upstream smoke:

- Root: `/data/ziheng/moe_infinity_fgo_runs/source_only_available_models_smoke_v1`

| Model | Upstream source workflow | Prefetch calls | Result |
| --- | --- | ---: | --- |
| `DeepSeek-V2-Lite` | succeeds | `0` | valid offloading baseline, not activation-aware prefetch baseline |
| `Qwen1.5-MoE-A2.7B-Chat` | fails | `0` | unsupported architecture in upstream source-only loader |
| `OLMoE-1B-7B-0924` | fails | `0` | tokenizer/API incompatibility in current upstream env |

## Feasibility of True Upstream Prefetch Models

The only active upstream prefetch wrappers found here are Arctic and Grok.
Those are not practical near-term baselines on the current server:

- Snowflake Arctic is a 480B-total MoE with about 17B active parameters and its
  Hugging Face page recommends an 8xH100-class setup.
- Grok-1 is a 314B-parameter MoE and its Hugging Face page says multi-GPU
  hardware is required.
- The current server has two A800 80GB GPUs and about 84GB free on `/data`;
  downloading and offloading either model is not a reasonable immediate step.

References:

- `https://huggingface.co/Snowflake/snowflake-arctic-instruct`
- `https://huggingface.co/xai-org/grok-1`

## Current Baseline Conclusion

For the currently available models, open-source MoE-Infinity source-only
baseline is an offloading/cache baseline, not an activation-aware prefetch
baseline. Any claim about improving "MoE-Infinity prefetch" must first either:

1. use a source workflow and model wrapper that actually calls upstream
   `ExpertPrefetcher.prefetch_experts()`, or
2. explicitly frame the comparison as an FGO/runtime mechanism study rather
   than an upstream prefetch comparison.

The defensible next research path is therefore not "beat upstream prefetch" on
DeepSeek/Qwen. It is to make a clean systems claim around expert paging
semantics: separating source-only offloading baseline, candidate retention,
actual H2D prefetch transfer, admission/progress, and lifecycle conversion.
