# Baseline Reproduction Requirements

Date: 2026-04-30

## Principle

MoE-Infinity upstream baselines must be reproduced through the upstream source
workflow first. A valid upstream baseline uses the original public API or
examples, for example:

```python
from moe_infinity import MoE

model = MoE(checkpoint, config)
output_ids = model.generate(input_ids)
```

Custom decode loops, manual KV-cache handling, manually inserted prefetch calls,
or modified routing/controller hooks are not valid upstream baselines.

## Evidence Labels

Use these labels in future experiment summaries:

- `source-only baseline`: original upstream API/script path. This is the only
  class that may appear in baseline performance tables.
- `environment failure`: original source path failed because of dependency,
  runtime, CUDA, or model API mismatch. This is workflow evidence only.
- `diagnostic wrapper`: custom decode/prefetch/control wrapper. This can be used
  to locate bugs, but must not be reported as upstream performance.

## Required Baseline Checks

Before comparing any modified runtime against MoE-Infinity:

1. Record the upstream repository path, commit, Python environment, `torch`,
   `transformers`, and CUDA versions.
2. Use upstream `MoE.generate()` or upstream example scripts. Do not replace
   generation with a hand-written loop.
3. Report whether the target model's source wrapper actually calls
   `ExpertPrefetcher.prefetch_experts()`.
4. If `prefetch=True` is set in config, verify that it changes source behavior;
   otherwise report it as a no-op for that model path.
5. If source-only generation fails, fix the environment or use a compatible
   upstream commit before building diagnostic wrappers.

## What Previous Manual Drivers Mean

Previous manual decode drivers are useful only for debugging dependency and KV
issues. They are not valid upstream baselines because they replace the original
generation workflow. Any result produced by such a driver must be labeled
`diagnostic wrapper`.
