from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_infinity.utils.qwen_trace_builder import (
    DEFAULT_SEED,
    DEFAULT_TRACE_LENGTH,
    build_trace_bundle,
    count_chat_tokens,
    filter_prompts_by_token_budget,
    load_finemoe_lmsys_prompts,
    write_trace_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dataset-derived Qwen benchmark traces from FineMoE's LMSYS prompt sample."
    )
    parser.add_argument(
        "--source-json",
        default=str(
            Path(__file__).resolve().parents[2]
            / "offloading_refs"
            / "FineMoE-EuroSys26"
            / "demo"
            / "states"
            / "lmsys-chat-1m~eval_prompts.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "traces" / "qwen"),
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen1.5-MoE-A2.7B-Chat",
    )
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trace-length", type=int, default=DEFAULT_TRACE_LENGTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_finemoe_lmsys_prompts(args.source_json)
    safe_prompt_count = len(prompts)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    prompts = filter_prompts_by_token_budget(
        prompts,
        token_counter=lambda prompt: count_chat_tokens(tokenizer, prompt),
        max_input_length=args.max_input_length,
    )
    bundle = build_trace_bundle(
        prompts,
        seed=args.seed,
        target_len=args.trace_length,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in bundle.items():
        write_trace_jsonl(output_dir / f"{name}.jsonl", rows)

    metadata = {
        "source_json": str(Path(args.source_json).resolve()),
        "model_path": args.model_path,
        "max_input_length": args.max_input_length,
        "seed": args.seed,
        "trace_length": args.trace_length,
        "safe_prompt_count": safe_prompt_count,
        "token_budget_prompt_count": len(prompts),
        "trace_names": sorted(bundle),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
