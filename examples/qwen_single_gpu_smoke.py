from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from moe_infinity import MoE
from moe_infinity.utils.qwen_smoke import (
    build_qwen_smoke_config,
    dispatcher_stats_dict,
    make_fresh_offload_path,
    visible_cuda_devices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-visible-GPU Qwen MoE-Infinity smoke."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--offload-root", required=True)
    parser.add_argument(
        "--phase",
        default="baseline",
        choices=["baseline", "history", "backbone"],
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--device-memory-ratio", type=float, default=0.6)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--library-capacity", type=int, default=8)
    parser.add_argument("--library-metric", default="cosine")
    parser.add_argument("--library-admission", default="diversity_aware")
    parser.add_argument("--backbone-topk", type=int, default=8)
    parser.add_argument("--summary-json")
    parser.add_argument(
        "--require-single-visible-gpu",
        action="store_true",
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke.")

    visible_count = torch.cuda.device_count()
    if args.require_single_visible_gpu and visible_count != 1:
        raise RuntimeError(
            f"Expected exactly 1 visible CUDA device, found {visible_count}. "
            "Use CUDA_VISIBLE_DEVICES to isolate a single GPU."
        )

    device = torch.device("cuda:0")
    offload_path = make_fresh_offload_path(args.offload_root, phase=args.phase)
    config = build_qwen_smoke_config(
        phase=args.phase,
        offload_path=offload_path,
        device_memory_ratio=args.device_memory_ratio,
        num_threads=args.num_threads,
        library_capacity=args.library_capacity,
        library_metric=args.library_metric,
        library_admission=args.library_admission,
        backbone_topk=args.backbone_topk,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    model = MoE(args.model_path, config)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Return exactly one short word: hello"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    dispatcher = model.engine.expert_dispatcher
    policy = model.engine.offloading_policy
    runs = []
    for step in range(args.steps):
        if hasattr(dispatcher, "reset_runtime_stats"):
            dispatcher.reset_runtime_stats()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        dispatcher_stats = {}
        if hasattr(dispatcher, "get_runtime_stats"):
            dispatcher_stats = dispatcher_stats_dict(
                dispatcher.get_runtime_stats()
            )
        libsize = policy.library_size() if policy is not None else -1
        libstats = policy.library_stats() if policy is not None else {}
        run = {
            "step": step,
            "library_size": libsize,
            "library_stats": libstats,
            "dispatcher_stats": dispatcher_stats,
            "text_prefix": text[:200],
        }
        runs.append(run)
        print(json.dumps(run, ensure_ascii=True), flush=True)

    summary = {
        "phase": args.phase,
        "model_path": args.model_path,
        "offload_path": offload_path,
        "config": config,
        "visible_cuda": visible_cuda_devices(
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            (torch.cuda.get_device_name(i) for i in range(visible_count)),
        ),
        "runs": runs,
    }
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
