from __future__ import annotations

__all__ = ["OffloadEngine"]


def __getattr__(name: str):
    if name == "OffloadEngine":
        from .model_offload import OffloadEngine

        return OffloadEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
