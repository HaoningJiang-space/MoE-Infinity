from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence


DEFAULT_TRACE_LENGTH = 34
DEFAULT_SEED = 42
MAX_PROMPT_CHARS = 1200

UNSAFE_PATTERN = re.compile(
    r"("
    r"nsfw|sexual|sex\b|porn|violent|violence|hate|offensive|seductive|horny|"
    r"vulgar|perverted|disgusting|immoral|aggressive|insult|club penguin|"
    r"genderswap|lewd|smut|erotic|exception to ai|toxic commentary|trick people into signing|"
    r"joystick|tongue|breast|swallow|euphoria|spray|shaft\b|balls in a sack|"
    r"roleplay|magic liquid|fetish|salty"
    r")",
    re.IGNORECASE,
)

CATEGORY_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "coding": re.compile(
        r"(\bpython\b|\bsql\b|\bbash\b|\bcode\b|\bfunction\b|\bbug\b|"
        r"\balgorithm\b|\bjava\b|\bjavascript\b|\bc\+\+\b|\bregex\b|"
        r"\bjson\b|\bcsv\b|\bapi\b|\bpandas\b|\bsqlite\b|\bprogramming\b|pip install)",
        re.IGNORECASE,
    ),
    "reasoning": re.compile(
        r"(compute|percentage|think step by step|determine|state which step|"
        r"evaluate|classif|logic|math|reasoning)",
        re.IGNORECASE,
    ),
    "writing": re.compile(
        r"(summarize|summary|improve the wording|reformuler|rewrite|translate|"
        r"manual|cms|email|report|korean|deutsch|french|german)",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class SourcePrompt:
    prompt: str
    category: str


def _normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_finemoe_lmsys_prompts(
    source_path: str | Path,
) -> List[SourcePrompt]:
    path = Path(source_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts: List[SourcePrompt] = []
    for item in payload:
        raw_prompt = item.get("prompt", "")
        if not isinstance(raw_prompt, str):
            continue
        prompt = _normalize_prompt(raw_prompt)
        if not prompt:
            continue
        if len(prompt) > MAX_PROMPT_CHARS:
            continue
        if UNSAFE_PATTERN.search(prompt):
            continue
        category = categorize_prompt(prompt)
        prompts.append(SourcePrompt(prompt=prompt, category=category))
    return prompts


def categorize_prompt(prompt: str) -> str:
    for name, pattern in CATEGORY_PATTERNS.items():
        if pattern.search(prompt):
            return name
    return "other"


def make_chat_messages(prompt: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def count_chat_tokens(tokenizer: object, prompt: str) -> int:
    rendered = tokenizer.apply_chat_template(
        make_chat_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, truncation=False)
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def filter_prompts_by_token_budget(
    prompts: Sequence[SourcePrompt],
    *,
    token_counter: Callable[[str], int],
    max_input_length: int,
) -> List[SourcePrompt]:
    filtered: List[SourcePrompt] = []
    for item in prompts:
        if token_counter(item.prompt) <= max_input_length:
            filtered.append(item)
    return filtered


def _dedup_preserve_order(prompts: Iterable[SourcePrompt]) -> List[SourcePrompt]:
    seen = set()
    result: List[SourcePrompt] = []
    for item in prompts:
        if item.prompt in seen:
            continue
        seen.add(item.prompt)
        result.append(item)
    return result


def _cycle_to_length(items: Sequence[SourcePrompt], target_len: int) -> List[SourcePrompt]:
    if not items:
        raise ValueError("Cannot cycle an empty prompt list")
    result: List[SourcePrompt] = []
    index = 0
    while len(result) < target_len:
        result.append(items[index % len(items)])
        index += 1
    return result


def build_stationary_prompts(
    prompts: Sequence[SourcePrompt],
    *,
    target_len: int = DEFAULT_TRACE_LENGTH,
    seed: int = DEFAULT_SEED,
) -> List[SourcePrompt]:
    rng = random.Random(seed)
    preferred = [
        item for item in prompts if item.category in {"coding", "reasoning", "writing"}
    ]
    preferred = _dedup_preserve_order(preferred)
    rng.shuffle(preferred)
    if len(preferred) >= target_len:
        return preferred[:target_len]
    return _cycle_to_length(preferred, target_len)


def build_mixed_prompts(
    prompts: Sequence[SourcePrompt],
    *,
    target_len: int = DEFAULT_TRACE_LENGTH,
    seed: int = DEFAULT_SEED,
) -> List[SourcePrompt]:
    rng = random.Random(seed)
    by_category: Dict[str, List[SourcePrompt]] = defaultdict(list)
    for item in _dedup_preserve_order(prompts):
        by_category[item.category].append(item)
    for values in by_category.values():
        rng.shuffle(values)

    category_order = ["coding", "reasoning", "writing", "other"]
    result: List[SourcePrompt] = []
    positions = {name: 0 for name in category_order}
    while len(result) < target_len:
        made_progress = False
        for category in category_order:
            bucket = by_category.get(category, [])
            if positions[category] >= len(bucket):
                continue
            result.append(bucket[positions[category]])
            positions[category] += 1
            made_progress = True
            if len(result) >= target_len:
                break
        if made_progress:
            continue
        fallback = _dedup_preserve_order(prompts)
        rng.shuffle(fallback)
        result.extend(_cycle_to_length(fallback, target_len - len(result)))
    return result[:target_len]


def build_recurrence_heavy_prompts(
    prompts: Sequence[SourcePrompt],
    *,
    target_len: int = DEFAULT_TRACE_LENGTH,
    seed: int = DEFAULT_SEED,
) -> List[SourcePrompt]:
    rng = random.Random(seed)
    coding = [item for item in _dedup_preserve_order(prompts) if item.category == "coding"]
    if len(coding) < 4:
        raise ValueError("Need at least 4 safe coding prompts for recurrence-heavy trace")
    rng.shuffle(coding)
    base = coding[:4]
    result: List[SourcePrompt] = []
    while len(result) < target_len:
        for item in base:
            result.append(item)
            if len(result) >= target_len:
                break
    return result


def serialize_trace_items(
    prompts: Sequence[SourcePrompt],
    *,
    trace_prefix: str,
) -> List[Dict[str, object]]:
    rows = []
    for index, item in enumerate(prompts, start=1):
        rows.append(
            {
                "request_id": f"{trace_prefix}-{index:03d}",
                "tag": item.category,
                "messages": make_chat_messages(item.prompt),
            }
        )
    return rows


def write_trace_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False))
            handle.write("\n")


def build_trace_bundle(
    prompts: Sequence[SourcePrompt],
    *,
    seed: int = DEFAULT_SEED,
    target_len: int = DEFAULT_TRACE_LENGTH,
) -> Dict[str, List[Dict[str, object]]]:
    stationary = serialize_trace_items(
        build_stationary_prompts(prompts, target_len=target_len, seed=seed),
        trace_prefix="stationary",
    )
    mixed = serialize_trace_items(
        build_mixed_prompts(prompts, target_len=target_len, seed=seed),
        trace_prefix="mixed",
    )
    recurrence = serialize_trace_items(
        build_recurrence_heavy_prompts(prompts, target_len=target_len, seed=seed),
        trace_prefix="recur",
    )
    return {
        "stationary": stationary,
        "mixed": mixed,
        "recurrence_heavy": recurrence,
    }
