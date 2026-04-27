# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Lazy model exports.

Most unit tests only need lightweight helpers. Importing every model eagerly
pulls in version-sensitive Transformers modules, which breaks collection in
minimal environments. Keep the public surface the same, but resolve imports on
first use.
"""

from __future__ import annotations

__all__ = [
    "ArcticConfig",
    "DeepseekMoEBlock",
    "Qwen2MoEBlock",
    "Qwen3MoEBlock",
    "SyncArcticMoeBlock",
    "SyncGrokMoeBlock",
    "SyncMixtralSparseMoeBlock",
    "SyncNllbMoeSparseMLP",
    "SyncSwitchTransformersSparseMLP",
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_deepseek",
    "rotate_half",
]


def __getattr__(name: str):
    if name in {"ArcticConfig", "SyncArcticMoeBlock"}:
        from .arctic import ArcticConfig, SyncArcticMoeBlock

        return {
            "ArcticConfig": ArcticConfig,
            "SyncArcticMoeBlock": SyncArcticMoeBlock,
        }[name]
    if name == "DeepseekMoEBlock":
        from .deepseek import DeepseekMoEBlock

        return DeepseekMoEBlock
    if name == "SyncGrokMoeBlock":
        from .grok import SyncGrokMoeBlock

        return SyncGrokMoeBlock
    if name == "SyncMixtralSparseMoeBlock":
        from .mixtral import SyncMixtralSparseMoeBlock

        return SyncMixtralSparseMoeBlock
    if name in {
        "apply_rotary_pos_emb",
        "apply_rotary_pos_emb_deepseek",
        "rotate_half",
    }:
        from .model_utils import (
            apply_rotary_pos_emb,
            apply_rotary_pos_emb_deepseek,
            rotate_half,
        )

        return {
            "apply_rotary_pos_emb": apply_rotary_pos_emb,
            "apply_rotary_pos_emb_deepseek": apply_rotary_pos_emb_deepseek,
            "rotate_half": rotate_half,
        }[name]
    if name == "SyncNllbMoeSparseMLP":
        from .nllb_moe import SyncNllbMoeSparseMLP

        return SyncNllbMoeSparseMLP
    if name in {"Qwen2MoEBlock", "Qwen3MoEBlock"}:
        from .qwen import Qwen2MoEBlock, Qwen3MoEBlock

        return {
            "Qwen2MoEBlock": Qwen2MoEBlock,
            "Qwen3MoEBlock": Qwen3MoEBlock,
        }[name]
    if name == "SyncSwitchTransformersSparseMLP":
        from .switch_transformers import SyncSwitchTransformersSparseMLP

        return SyncSwitchTransformersSparseMLP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
