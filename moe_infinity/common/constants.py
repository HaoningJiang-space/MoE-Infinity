from importlib import import_module

from transformers import (
    MixtralForCausalLM,
    NllbMoeForConditionalGeneration,
    OPTForCausalLM,
    PretrainedConfig,
    Qwen2MoeForCausalLM,
    Qwen3MoeForCausalLM,
    SwitchTransformersForConditionalGeneration,
)


def _optional_import(module_name, attr_name):
    try:
        module = import_module(module_name)
        return getattr(module, attr_name)
    except Exception:
        return None


ArcticForCausalLM = _optional_import(
    "moe_infinity.models.modeling_arctic",
    "ArcticForCausalLM",
)
DeepseekV2ForCausalLM = _optional_import(
    "moe_infinity.models.modeling_deepseek_v2",
    "DeepseekV2ForCausalLM",
)
DeepseekV3ForCausalLM = _optional_import(
    "moe_infinity.models.modeling_deepseek_v3",
    "DeepseekV3ForCausalLM",
)
Grok1ModelForCausalLM = _optional_import(
    "moe_infinity.models.modeling_grok.modeling_grok1",
    "Grok1ModelForCausalLM",
)

MODEL_MAPPING_NAMES = {
    "switch": SwitchTransformersForConditionalGeneration,
    "nllb": NllbMoeForConditionalGeneration,
    "mixtral": MixtralForCausalLM,
    "opt": OPTForCausalLM,
    "grok": Grok1ModelForCausalLM,
    "arctic": ArcticForCausalLM,
    "deepseek": DeepseekV2ForCausalLM,
    "deepseek_v3": DeepseekV3ForCausalLM,
    "qwen2": Qwen2MoeForCausalLM,
    "qwen3": Qwen3MoeForCausalLM,
}

MODEL_MAPPING_TYPES = {
    "switch": 0,
    "nllb": 2,
    "mixtral": 4,
    "grok": 4,
    "arctic": 4,
    "deepseek": 5,
    "deepseek_v3": 5,
    "qwen2": 5,
    "qwen3": 5,
}


def parse_expert_type(config: PretrainedConfig) -> int:
    architecture = config.architectures[0].lower()
    arch = None
    for supp_arch in MODEL_MAPPING_NAMES:
        if supp_arch in architecture:
            arch = supp_arch
            break
    if arch is None:
        raise RuntimeError(
            f"The `load_checkpoint_and_dispatch` function does not support the architecture {architecture}. "
            f"Please provide a model that is supported by the function. "
            f"Supported architectures are {list(MODEL_MAPPING_NAMES.keys())}."
        )

    return MODEL_MAPPING_TYPES[arch]
