from __future__ import annotations

import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import source_only_fair_matrix_v1 as base


ROOT = Path(
    os.environ.get(
        "SOURCE_ONLY_FAIR_MATRIX_ROOT",
        "/data/ziheng/moe_infinity_fgo_runs/source_only_fair_matrix_v2",
    )
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_text(
    cmd: List[str], *, cwd: Path | None = None, env: Dict[str, str] | None = None
) -> str:
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _repo_metadata(repo: Path) -> Dict[str, Any]:
    status = _run_text(["git", "status", "--porcelain"], cwd=repo)
    return {
        "path": str(repo),
        "commit": _run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "branch": _run_text(["git", "branch", "--show-current"], cwd=repo),
        "remote": _run_text(["git", "remote", "-v"], cwd=repo),
        "dirty": bool(status.strip()),
        "status_short": status,
    }


def _python_metadata(repo: Path, python: Path) -> Dict[str, Any]:
    script = (
        "import json, sys\n"
        "payload={'python': sys.version, 'executable': sys.executable}\n"
        "try:\n"
        " import torch\n"
        " payload['torch']=torch.__version__\n"
        " payload['torch_cuda']=getattr(torch.version, 'cuda', None)\n"
        " payload['cuda_available']=bool(torch.cuda.is_available())\n"
        "except Exception as exc:\n"
        " payload['torch_error']=type(exc).__name__+': '+str(exc)\n"
        "try:\n"
        " import transformers\n"
        " payload['transformers']=transformers.__version__\n"
        "except Exception as exc:\n"
        " payload['transformers_error']=type(exc).__name__+': '+str(exc)\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    output = _run_text(
        [str(python), "-c", script],
        cwd=repo,
        env=base._env(repo, python),
    )
    try:
        return json.loads(output)
    except Exception:
        return {"metadata_error": output}


def _trace_metadata(trace_file: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "path": str(trace_file),
        "exists": trace_file.exists(),
        "line_count": 0,
        "json_records": 0,
        "message_records": 0,
        "prompt_records": 0,
        "text_char_p50": 0,
        "text_char_p95": 0,
    }
    if not trace_file.exists():
        return payload

    lengths: List[int] = []
    with trace_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload["line_count"] += 1
            try:
                record = json.loads(line)
            except Exception:
                continue
            payload["json_records"] += 1
            if isinstance(record.get("messages"), list):
                payload["message_records"] += 1
                text = " ".join(
                    str(item.get("content", ""))
                    for item in record["messages"]
                    if isinstance(item, dict)
                )
            else:
                text = ""
                for key in ("prompt", "text", "input", "content"):
                    if isinstance(record.get(key), str):
                        text = record[key]
                        payload["prompt_records"] += 1
                        break
            if text:
                lengths.append(len(text))
    if lengths:
        lengths = sorted(lengths)
        payload["text_char_p50"] = lengths[len(lengths) // 2]
        payload["text_char_p95"] = lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))]
    return payload


def _prefetch_call_count(result: Dict[str, Any]) -> int:
    return base._prefetch_call_count(result)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _input_token_stats(items: List[Dict[str, Any]]) -> Dict[str, float]:
    tokens: List[float] = []
    generated: List[float] = []
    for item in items:
        for record in item.get("records", []) or []:
            if isinstance(record, dict):
                tokens.append(float(record.get("input_tokens") or 0.0))
                generated.append(float(record.get("generated_tokens") or 0.0))
    if not tokens:
        return {
            "input_tokens_mean": 0.0,
            "input_tokens_p50": 0.0,
            "input_tokens_p95": 0.0,
            "generated_tokens_mean": 0.0,
        }
    return {
        "input_tokens_mean": statistics.mean(tokens),
        "input_tokens_p50": statistics.median(tokens),
        "input_tokens_p95": sorted(tokens)[min(len(tokens) - 1, int(len(tokens) * 0.95))],
        "generated_tokens_mean": statistics.mean(generated) if generated else 0.0,
    }


def _summarize_v2(results: List[Dict[str, Any]]) -> None:
    analysis = ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    repos = {
        "fgo": _repo_metadata(base.FGO_REPO),
        "upstream": _repo_metadata(base.UPSTREAM_REPO),
    }
    pythons = {
        "fgo": _python_metadata(base.FGO_REPO, base.FGO_PYTHON),
        "upstream": _python_metadata(base.UPSTREAM_REPO, base.UPSTREAM_PYTHON),
    }
    trace = _trace_metadata(base.TRACE_FILE)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result.get("case", "")), []).append(result)

    rows = []
    for case_name, items in grouped.items():
        ok = [
            item
            for item in items
            if item.get("success") and int(item.get("returncode", 1)) == 0
        ]
        repo = str(items[0].get("repo", ""))
        token_stats = _input_token_stats(ok)
        prefetch_calls = sum(_prefetch_call_count(item) for item in items)
        failures = [item for item in items if item not in ok]
        rows.append(
            {
                "case": case_name,
                "repo": repo,
                "workflow": "MoE(...).generate",
                "label": "source-only baseline" if ok else "environment failure",
                "repeats_ok": len(ok),
                "repeats_total": len(items),
                "mean_tpot_ms": _mean(
                    float(item.get("decode_tpot_ms") or 0.0) for item in ok
                ),
                "mean_tok_s": _mean(
                    float(item.get("decode_tokens_per_second") or 0.0) for item in ok
                ),
                "mean_wall_s": _mean(float(item.get("total_wall_s") or 0.0) for item in ok),
                "mean_setup_s": _mean(float(item.get("setup_s") or 0.0) for item in ok),
                "prefetch_calls": prefetch_calls,
                "prefetch_triggered": prefetch_calls > 0,
                "failure_type": str(failures[0].get("error_type", "")) if failures else "",
                "failure": str(failures[0].get("error", "")) if failures else "",
                **token_stats,
            }
        )

    payload = {
        "timestamp_utc": _now(),
        "version": "source_only_fair_matrix_v2",
        "root": str(ROOT),
        "model": str(base.MODEL),
        "trace_file": str(base.TRACE_FILE),
        "cuda_visible_devices": base.CUDA_VISIBLE_DEVICES,
        "repos": repos,
        "pythons": pythons,
        "trace": trace,
        "rows": rows,
        "results": results,
    }
    (analysis / "source_only_fair_matrix_v2.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Source-Only Fair Matrix V2",
        "",
        "All valid rows use the original public workflow `MoE(...); model.generate(...)`.",
        "Counter rows only wrap prefetch methods for observation and do not manually call prefetch.",
        "",
        f"- Model: `{base.MODEL}`",
        f"- Trace file: `{base.TRACE_FILE}`",
        f"- Trace JSON records: `{trace.get('json_records', 0)}`",
        f"- CUDA_VISIBLE_DEVICES: `{base.CUDA_VISIBLE_DEVICES}`",
        "",
        "## Environment",
        "",
        "| repo | commit | dirty | python | torch | transformers |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for repo in ("upstream", "fgo"):
        lines.append(
            "| {repo} | `{commit}` | {dirty} | `{python}` | `{torch}` | `{transformers}` |".format(
                repo=repo,
                commit=str(repos[repo].get("commit", ""))[:12],
                dirty=str(repos[repo].get("dirty", "")),
                python=str(pythons[repo].get("executable", "")),
                torch=str(pythons[repo].get("torch", pythons[repo].get("torch_error", ""))),
                transformers=str(
                    pythons[repo].get(
                        "transformers", pythons[repo].get("transformers_error", "")
                    )
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| case | repo | ok/total | TPOT ms | tok/s | wall s | setup s | input p50/p95 | prefetch calls | triggered | label | failure |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |",
        ]
    )
    for row in rows:
        failure = row["failure"]
        if len(failure) > 80:
            failure = failure[:77] + "..."
        lines.append(
            "| {case} | {repo} | {ok}/{total} | {tpot:.3f} | {tps:.3f} | {wall:.3f} | {setup:.3f} | {p50:.0f}/{p95:.0f} | {calls} | {triggered} | {label} | {failure} |".format(
                case=row["case"],
                repo=row["repo"],
                ok=row["repeats_ok"],
                total=row["repeats_total"],
                tpot=row["mean_tpot_ms"],
                tps=row["mean_tok_s"],
                wall=row["mean_wall_s"],
                setup=row["mean_setup_s"],
                p50=row["input_tokens_p50"],
                p95=row["input_tokens_p95"],
                calls=row["prefetch_calls"],
                triggered=str(row["prefetch_triggered"]),
                label=row["label"],
                failure=failure.replace("|", "\\|"),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Rows labeled `source-only baseline` are valid baseline rows.",
            "- Rows labeled `environment failure` are workflow evidence only, not performance numbers.",
            "- Zero prefetch calls means this model/workflow did not exercise the upstream activation-aware prefetch path.",
            "- Custom decode, manual KV propagation, and manual prefetch are intentionally excluded from this matrix.",
        ]
    )
    (analysis / "source_only_fair_matrix_v2.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    base.ROOT = ROOT
    args = base.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    runner = ROOT / "source_only_generate_runner.py"
    base._write_runner(runner)
    cases = base._selected(
        args.cases,
        "SOURCE_ONLY_FAIR_MATRIX_CASES",
        ["upstream_plain", "upstream_counter", "fgo_plain", "fgo_counter"],
    )
    results: List[Dict[str, Any]] = []
    for repeat in range(args.repeats):
        for case_name in cases:
            results.append(
                base._run_case(
                    case_name=case_name,
                    case=base.CASES[case_name],
                    repeat=repeat,
                    runner=runner,
                    args=args,
                )
            )
    base._summarize(results)
    _summarize_v2(results)
    print(ROOT / "analysis" / "source_only_fair_matrix_v2.md")


if __name__ == "__main__":
    main()
