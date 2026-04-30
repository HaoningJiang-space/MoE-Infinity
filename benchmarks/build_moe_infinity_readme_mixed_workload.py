from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import signal
import statistics
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from datasets import get_dataset_config_names, load_dataset
from transformers import AutoTokenizer


README_DATASETS = (
    "THUDM/LongBench",
    "openai/gsm8k",
    "Muennighoff/flan",
    "tasksource/bigbench",
    "lukaemon/mmlu",
)

LONG_BENCH_ZIP_REPO_PATH = "datasets/THUDM/LongBench/resolve/main/data.zip"
LONG_BENCH_DEFAULT_ZIP_URL = f"https://huggingface.co/{LONG_BENCH_ZIP_REPO_PATH}"

BIGBENCH_EXCLUDE = {
    "simple_arithmetic_json_multiple_choice",
    "simple_arithmetic_multiple_targets_json",
    "cifar10_classification",
}


class WorkloadBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    source_dataset: str
    source_config: str
    source_split: str
    prompt: str
    raw_fields: Mapping[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize(value)
    return ""


def _format_choices(choices: Any) -> str:
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        return ""
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for index, choice in enumerate(choices):
        if index >= len(labels):
            break
        lines.append(f"({labels[index]}) {choice}")
    return "\n".join(lines)


def _prompt_from_record(dataset_name: str, record: Mapping[str, Any]) -> str:
    lower_name = dataset_name.lower()
    if "gsm8k" in lower_name:
        return _first_text(record, ("question", "input", "prompt"))
    if "mmlu" in lower_name:
        question = _first_text(record, ("question", "input", "prompt"))
        choices = _format_choices(record.get("choices"))
        return f"{question}\nOptions:\n{choices}" if choices else question
    if "bigbench" in lower_name:
        prompt = _first_text(record, ("inputs", "input", "question", "prompt"))
        if prompt:
            return prompt
    if "flan" in lower_name:
        prompt = _first_text(record, ("inputs", "input", "question", "source", "prompt"))
        if prompt:
            return prompt
    if "longbench" in lower_name:
        prompt = _first_text(record, ("input", "question", "query", "prompt"))
        context = _first_text(record, ("context", "article", "passage"))
        if context and prompt:
            return f"{context}\n\nQuestion: {prompt}"
        if prompt:
            return prompt
    return _first_text(record, ("input", "inputs", "question", "prompt", "text"))


def _chat_messages(prompt: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _chat_token_count(tokenizer: Any, prompt: str) -> int:
    rendered = tokenizer.apply_chat_template(
        _chat_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, truncation=False)
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _truncate_prompt_to_budget(tokenizer: Any, prompt: str, max_input_length: int) -> str:
    prompt = _normalize(prompt)
    if _chat_token_count(tokenizer, prompt) <= max_input_length:
        return prompt
    low = 0
    high = len(prompt)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = prompt[:mid].rstrip()
        if not candidate:
            high = mid - 1
            continue
        if _chat_token_count(tokenizer, candidate) <= max_input_length:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    if not best:
        raise WorkloadBuildError("Could not truncate prompt to fit token budget")
    return best


def _split_candidates() -> Sequence[str]:
    return ("validation", "test", "train")


def _load_split(dataset_name: str, config: str | None, split: str, cache_dir: str) -> Any:
    if config:
        return load_dataset(
            dataset_name,
            config,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    return load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )


def _load_first_available_split(
    dataset_name: str,
    config: str | None,
    cache_dir: str,
) -> tuple[Any, str]:
    errors = []
    for split in _split_candidates():
        try:
            return _load_split(dataset_name, config, split, cache_dir), split
        except Exception as exc:  # noqa: BLE001 - capture provenance for strict failure.
            errors.append(f"{split}: {type(exc).__name__}: {exc}")
    config_text = config if config is not None else "<default>"
    raise WorkloadBuildError(
        f"Failed to load {dataset_name}/{config_text}; tried splits "
        f"{list(_split_candidates())}; errors={errors}"
    )


def _sample_from_loaded_dataset(
    *,
    dataset_name: str,
    config: str,
    split: str,
    dataset: Any,
    sample_count: int,
    rng: random.Random,
) -> List[Candidate]:
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    result: List[Candidate] = []
    for index in indices:
        record = dict(dataset[index])
        prompt = _prompt_from_record(dataset_name, record)
        if not prompt:
            continue
        result.append(
            Candidate(
                source_dataset=dataset_name,
                source_config=config,
                source_split=split,
                prompt=prompt,
                raw_fields=_jsonable(record),
            )
        )
        if len(result) >= sample_count:
            break
    return result


def _load_config_names(dataset_name: str) -> List[str]:
    configs = list(get_dataset_config_names(dataset_name, trust_remote_code=True))
    if dataset_name == "tasksource/bigbench":
        configs = [name for name in configs if name not in BIGBENCH_EXCLUDE]
    return configs


def _mirror_url(repo_path: str, default_url: str) -> str:
    override = os.environ.get("MOE_README_LONGBENCH_ZIP_URL", "").strip()
    if override:
        return override
    endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    if endpoint:
        return f"{endpoint}/{repo_path}"
    return default_url


def _download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    urllib.request.urlretrieve(url, temporary)  # noqa: S310 - benchmark data URL.
    temporary.replace(target)


def _ensure_longbench_zip(cache_dir: str) -> tuple[Path, str]:
    url = _mirror_url(LONG_BENCH_ZIP_REPO_PATH, LONG_BENCH_DEFAULT_ZIP_URL)
    zip_path = (
        Path(cache_dir)
        / "moe_readme_downloads"
        / "THUDM_LongBench"
        / "data.zip"
    )
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        _download_file(url, zip_path)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.testzip()
    except zipfile.BadZipFile:
        zip_path.unlink(missing_ok=True)
        _download_file(url, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.testzip()
    return zip_path, url


def _sample_longbench_from_zip(
    *,
    sample_count: int,
    cache_dir: str,
    rng: random.Random,
    load_timeout_s: int,
    config_overrides: Mapping[str, Sequence[str | None]],
) -> tuple[List[Candidate], Dict[str, Any]]:
    provenance: Dict[str, Any] = {
        "dataset_name": "THUDM/LongBench",
        "requested_count": sample_count,
        "configs_attempted": [],
        "errors": [],
        "loader": "data.zip",
    }
    result: List[Candidate] = []
    with _time_limit(load_timeout_s, "downloading LongBench data.zip"):
        zip_path, data_url = _ensure_longbench_zip(cache_dir)
    provenance["data_url"] = data_url
    provenance["zip_path"] = str(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        if "THUDM/LongBench" in config_overrides:
            configs = tuple(config_overrides["THUDM/LongBench"])
        else:
            configs = tuple(
                sorted(
                    Path(member).stem
                    for member in archive.namelist()
                    if member.startswith("data/") and member.endswith(".jsonl")
                )
            )
            configs = list(configs)
            rng.shuffle(configs)
        candidates_by_config: Dict[str, List[Candidate]] = {}
        for config in configs:
            if config is None:
                continue
            config_text = str(config)
            provenance["configs_attempted"].append(config_text)
            member = f"data/{config_text}.jsonl"
            try:
                with archive.open(member) as handle:
                    records = [json.loads(line) for line in handle if line.strip()]
                candidates: List[Candidate] = []
                for record in records:
                    prompt = _prompt_from_record("THUDM/LongBench", record)
                    if not prompt:
                        continue
                    candidates.append(
                        Candidate(
                            source_dataset="THUDM/LongBench",
                            source_config=config_text,
                            source_split="test",
                            prompt=prompt,
                            raw_fields=_jsonable(record),
                        )
                    )
                rng.shuffle(candidates)
                if candidates:
                    candidates_by_config[config_text] = candidates
            except Exception as exc:  # noqa: BLE001 - fail later with provenance.
                provenance["errors"].append(
                    {
                        "config": config_text,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        while len(result) < sample_count and candidates_by_config:
            made_progress = False
            for config_text in list(candidates_by_config):
                candidates = candidates_by_config[config_text]
                if not candidates:
                    del candidates_by_config[config_text]
                    continue
                result.append(candidates.pop())
                made_progress = True
                if len(result) >= sample_count:
                    break
            if not made_progress:
                break
    provenance["loaded_count"] = len(result)
    if len(result) < sample_count:
        raise WorkloadBuildError(
            f"THUDM/LongBench produced {len(result)} prompts, "
            f"requested {sample_count}; provenance={provenance}"
        )
    return result[:sample_count], provenance


def _split_config_arg(value: str) -> List[str | None]:
    configs: List[str | None] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        configs.append(None if item in {"<default>", "default", "none"} else item)
    return configs


@contextlib.contextmanager
def _time_limit(seconds: int, label: str) -> Any:
    if seconds <= 0:
        yield
        return

    def _raise_timeout(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise TimeoutError(f"Timed out after {seconds}s while {label}")

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _sample_dataset(
    *,
    dataset_name: str,
    sample_count: int,
    cache_dir: str,
    rng: random.Random,
    load_timeout_s: int,
    config_overrides: Mapping[str, Sequence[str | None]],
) -> tuple[List[Candidate], Dict[str, Any]]:
    if dataset_name == "THUDM/LongBench":
        return _sample_longbench_from_zip(
            sample_count=sample_count,
            cache_dir=cache_dir,
            rng=rng,
            load_timeout_s=load_timeout_s,
            config_overrides=config_overrides,
        )

    provenance: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "requested_count": sample_count,
        "configs_attempted": [],
        "errors": [],
    }
    result: List[Candidate] = []

    if dataset_name in config_overrides:
        configs = tuple(config_overrides[dataset_name])
    elif dataset_name == "openai/gsm8k":
        configs: Iterable[str | None] = ("main",)
    elif dataset_name == "lukaemon/mmlu":
        with _time_limit(load_timeout_s, f"loading configs for {dataset_name}"):
            configs = _load_config_names(dataset_name)
        if not configs:
            raise WorkloadBuildError(f"{dataset_name} returned no configs")
        configs = list(configs)
        rng.shuffle(configs)
    elif dataset_name in {"THUDM/LongBench", "Muennighoff/flan", "tasksource/bigbench"}:
        with _time_limit(load_timeout_s, f"loading configs for {dataset_name}"):
            configs = _load_config_names(dataset_name)
        if not configs:
            raise WorkloadBuildError(f"{dataset_name} returned no configs")
        configs = list(configs)
        rng.shuffle(configs)
    else:
        configs = (None,)

    for config in configs:
        if len(result) >= sample_count:
            break
        config_text = config if config is not None else "<default>"
        provenance["configs_attempted"].append(config_text)
        try:
            with _time_limit(
                load_timeout_s,
                f"loading dataset {dataset_name}/{config_text}",
            ):
                dataset, split = _load_first_available_split(
                    dataset_name,
                    config,
                    cache_dir,
                )
            per_config_target = max(1, math.ceil(sample_count / max(1, len(configs))))
            needed = min(per_config_target, sample_count - len(result))
            result.extend(
                _sample_from_loaded_dataset(
                    dataset_name=dataset_name,
                    config=config_text,
                    split=split,
                    dataset=dataset,
                    sample_count=needed,
                    rng=rng,
                )
            )
        except Exception as exc:  # noqa: BLE001 - fail later with provenance.
            provenance["errors"].append(
                {
                    "config": config_text,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    provenance["loaded_count"] = len(result)
    if len(result) < sample_count:
        raise WorkloadBuildError(
            f"{dataset_name} produced {len(result)} prompts, "
            f"requested {sample_count}; provenance={provenance}"
        )
    return result[:sample_count], provenance


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False))
            handle.write("\n")


def _summarize_lengths(lengths: Sequence[int]) -> Dict[str, float | int]:
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(lengths)

    def percentile(pct: float) -> float:
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
        return float(ordered[int(index)])

    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": float(statistics.mean(lengths)),
        "p50": percentile(50.0),
        "p95": percentile(95.0),
    }


def build_workload(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    dataset_rows: Dict[str, List[Dict[str, Any]]] = {}
    provenance: Dict[str, Any] = {
        "builder": Path(__file__).name,
        "strict_readme_datasets": list(README_DATASETS),
        "seed": args.seed,
        "per_dataset": args.per_dataset,
        "max_input_length": args.max_input_length,
        "model_path": args.model_path,
        "cache_dir": args.cache_dir,
        "load_timeout_s": args.load_timeout_s,
        "config_overrides": {
            dataset_name: [
                config if config is not None else "<default>"
                for config in configs
            ]
            for dataset_name, configs in _config_overrides(args).items()
        },
        "output_dir": str(Path(args.output_dir).resolve()),
        "datasets": {},
    }

    for dataset_name in README_DATASETS:
        candidates, dataset_provenance = _sample_dataset(
            dataset_name=dataset_name,
            sample_count=args.per_dataset,
            cache_dir=args.cache_dir,
            rng=random.Random(args.seed + sum(ord(ch) for ch in dataset_name)),
            load_timeout_s=args.load_timeout_s,
            config_overrides=_config_overrides(args),
        )
        rows: List[Dict[str, Any]] = []
        lengths: List[int] = []
        truncated_count = 0
        for local_index, candidate in enumerate(candidates, start=1):
            original_tokens = _chat_token_count(tokenizer, candidate.prompt)
            prompt = _truncate_prompt_to_budget(
                tokenizer,
                candidate.prompt,
                args.max_input_length,
            )
            final_tokens = _chat_token_count(tokenizer, prompt)
            truncated = final_tokens < original_tokens
            if truncated:
                truncated_count += 1
            lengths.append(final_tokens)
            key = dataset_name.lower().replace("/", "_").replace("-", "_")
            rows.append(
                {
                    "request_id": f"readme-{key}-{local_index:04d}",
                    "tag": key,
                    "source_dataset": candidate.source_dataset,
                    "source_config": candidate.source_config,
                    "source_split": candidate.source_split,
                    "messages": _chat_messages(prompt),
                    "input_tokens": final_tokens,
                    "original_input_tokens": original_tokens,
                    "truncated_to_max_input_length": truncated,
                    "raw_fields": candidate.raw_fields,
                }
            )
        dataset_provenance["final_count"] = len(rows)
        dataset_provenance["truncated_count"] = truncated_count
        dataset_provenance["input_token_summary"] = _summarize_lengths(lengths)
        provenance["datasets"][dataset_name] = dataset_provenance
        dataset_rows[dataset_name] = rows

    mixed_rows: List[Dict[str, Any]] = []
    for offset in range(args.per_dataset):
        for dataset_name in README_DATASETS:
            mixed_rows.append(dataset_rows[dataset_name][offset])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "mixed.jsonl", mixed_rows)
    provenance["total_count"] = len(mixed_rows)
    provenance["dataset_order"] = list(README_DATASETS)
    provenance["overall_input_token_summary"] = _summarize_lengths(
        [int(row["input_tokens"]) for row in mixed_rows]
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the strict README-aligned mixed workload for MoE-Infinity."
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).resolve().parent
            / "traces"
            / "moe_infinity_readme_mixed_v1"
        ),
    )
    parser.add_argument(
        "--model-path",
        default="/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat",
    )
    parser.add_argument("--cache-dir", default="/data/ziheng/hf_datasets")
    parser.add_argument("--per-dataset", type=int, default=64)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument(
        "--longbench-configs",
        default="discover",
        help="Comma-separated THUDM/LongBench configs; use 'discover' for all configs.",
    )
    parser.add_argument(
        "--flan-configs",
        default="<default>",
        help="Comma-separated Muennighoff/flan configs; '<default>' means no config.",
    )
    parser.add_argument(
        "--bigbench-configs",
        default="discover",
        help="Comma-separated tasksource/bigbench configs; use 'discover' for all configs.",
    )
    parser.add_argument(
        "--mmlu-configs",
        default="discover",
        help="Comma-separated lukaemon/mmlu configs; use 'discover' for all configs.",
    )
    parser.add_argument(
        "--load-timeout-s",
        type=int,
        default=120,
        help="Hard timeout for each dataset config/listing load; 0 disables timeout.",
    )
    return parser.parse_args()


def _config_overrides(args: argparse.Namespace) -> Dict[str, Sequence[str | None]]:
    overrides: Dict[str, Sequence[str | None]] = {
        "Muennighoff/flan": _split_config_arg(args.flan_configs),
    }
    if args.mmlu_configs.strip() != "discover":
        overrides["lukaemon/mmlu"] = _split_config_arg(args.mmlu_configs)
    if args.longbench_configs.strip() != "discover":
        overrides["THUDM/LongBench"] = _split_config_arg(args.longbench_configs)
    if args.bigbench_configs.strip() != "discover":
        overrides["tasksource/bigbench"] = _split_config_arg(args.bigbench_configs)
    return overrides


def main() -> None:
    args = parse_args()
    if args.per_dataset <= 0:
        raise ValueError("--per-dataset must be positive")
    try:
        build_workload(args)
    except Exception as exc:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "builder": Path(__file__).name,
            "strict_readme_datasets": list(README_DATASETS),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "output_dir": str(output_dir.resolve()),
            "per_dataset": args.per_dataset,
            "max_input_length": args.max_input_length,
            "seed": args.seed,
            "config_overrides": {
                dataset_name: [
                    config if config is not None else "<default>"
                    for config in configs
                ]
                for dataset_name, configs in _config_overrides(args).items()
            },
        }
        (output_dir / "build_failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
