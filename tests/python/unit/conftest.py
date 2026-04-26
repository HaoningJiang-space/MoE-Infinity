"""Unit-test conftest: make source tree importable and stub optional deps."""

import sys
from pathlib import Path
from unittest.mock import MagicMock


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stub_if_missing(name: str) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = MagicMock()


# nvtx: optional NVIDIA profiling library used in moe_infinity.models.*
_stub_if_missing("nvtx")

# Compiled CUDA extensions: may be absent in CPU-only test environments
_stub_if_missing("moe_infinity._store")
_stub_if_missing("moe_infinity._engine")
