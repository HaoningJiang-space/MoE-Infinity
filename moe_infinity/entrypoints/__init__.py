from __future__ import annotations

__all__ = ["MoE"]


def __getattr__(name: str):
    if name == "MoE":
        from .big_modeling import MoE

        return MoE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
