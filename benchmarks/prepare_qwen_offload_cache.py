from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import torch

from moe_infinity import MoE
from moe_infinity.utils.qwen_benchmark import build_qwen_benchmark_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a reusable read-mostly Qwen offload store for fast benchmark iteration."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-memory-ratio", type=float, default=0.30)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--library-capacity", type=int, default=32)
    parser.add_argument("--library-metric", default="cosine")
    parser.add_argument("--library-admission", default="diversity_aware")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _validate_existing(output_dir: Path) -> bool:
    return (output_dir / "archer_index").exists()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if _validate_existing(output_dir) and not args.force:
            print(f"offload cache already exists: {output_dir}")
            return
        if not args.force:
            raise FileExistsError(
                f"{output_dir} exists but does not look complete; pass --force to rebuild"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = build_qwen_benchmark_config(
        variant="on_demand",
        offload_path=str(output_dir),
        device_memory_ratio=args.device_memory_ratio,
        num_threads=args.num_threads,
        library_capacity=args.library_capacity,
        library_metric=args.library_metric,
        library_admission=args.library_admission,
    )

    started = time.perf_counter()
    model = MoE(args.model_path, config)
    elapsed_s = time.perf_counter() - started
    del model
    torch.cuda.empty_cache()

    if not _validate_existing(output_dir):
        raise RuntimeError(f"offload cache build did not create archer_index: {output_dir}")

    manifest = {
        "model_path": args.model_path,
        "output_dir": str(output_dir),
        "device_memory_ratio": args.device_memory_ratio,
        "num_threads": args.num_threads,
        "elapsed_s": elapsed_s,
        "note": (
            "This directory is intended for sequential fast iteration only. "
            "It reuses immutable expert-weight storage, not GPU residency or runtime state."
        ),
    }
    (output_dir / "offload_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
