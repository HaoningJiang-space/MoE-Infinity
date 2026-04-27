"""Top-level package exports with lazy imports.

Keeping these imports lazy avoids pulling in heavy model/runtime stacks when
unit tests only need lightweight utility or policy modules.
"""

from __future__ import annotations

__all__ = ["MoE", "OffloadEngine", "__version__"]

__version__ = "0.0.1"


def __getattr__(name: str):
    if name == "MoE":
        from moe_infinity.entrypoints import MoE

        return MoE
    if name == "OffloadEngine":
        from moe_infinity.runtime import OffloadEngine

        return OffloadEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
