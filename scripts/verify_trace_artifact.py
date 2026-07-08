"""Verify compact trace artifacts against committed CCF-A tables."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


TRACE_ROOT = Path("paper_artifacts/ccfa_trace_20260708")
MAIN_TASK_OUTCOMES = TRACE_ROOT / "main_task_outcomes.csv"
CYBER_TASK_OUTCOMES = TRACE_ROOT / "cyberseceval_task_outcomes.csv"
COMPLETION_CACHE_MANIFEST = TRACE_ROOT / "completion_cache_manifest.csv"
SOURCE_FILE_HASHES = TRACE_ROOT / "source_file_hashes.csv"
TRACE_MANIFEST = TRACE_ROOT / "trace_manifest.json"

MAIN_SEED_ROWS = Path("paper_artifacts/ccfa_20260707/main_seed_rows.csv")
CYBER_SEED_ROWS = Path("paper_artifacts/ccfa_20260708/ccfa_seed_rows.csv")


def main() -> None:
    report = verify_trace_artifact()
    print("CCF-A trace artifact verification passed.")
    print(json.dumps(report, indent=2))


def verify_trace_artifact() -> dict[str, object]:
    _require_files(
        [
            MAIN_TASK_OUTCOMES,
            CYBER_TASK_OUTCOMES,
            COMPLETION_CACHE_MANIFEST,
            SOURCE_FILE_HASHES,
            TRACE_MANIFEST,
            MAIN_SEED_ROWS,
            CYBER_SEED_ROWS,
        ]
    )
    main_report = _verify_one(MAIN_TASK_OUTCOMES, MAIN_SEED_ROWS, expected_seed_rows=375)
    cyber_report = _verify_one(CYBER_TASK_OUTCOMES, CYBER_SEED_ROWS, expected_seed_rows=60)
    cache_rows = _read_csv(COMPLETION_CACHE_MANIFEST)
    hash_rows = _read_csv(SOURCE_FILE_HASHES)
    if not cache_rows:
        raise SystemExit("empty completion cache manifest")
    if not hash_rows:
        raise SystemExit("empty source file hash manifest")
    return {
        "main": main_report,
        "cyberseceval": cyber_report,
        "completion_cache_manifest_rows": len(cache_rows),
        "source_file_hash_rows": len(hash_rows),
    }


def _verify_one(task_outcomes_path: Path, expected_seed_rows_path: Path, *, expected_seed_rows: int) -> dict[str, object]:
    task_rows = _read_csv(task_outcomes_path)
    expected_rows = _read_csv(expected_seed_rows_path)
    if len(expected_rows) < expected_seed_rows:
        raise SystemExit(
            f"{expected_seed_rows_path} has {len(expected_rows)} rows, expected at least {expected_seed_rows}"
        )
    actual = _aggregate_task_outcomes(task_rows)
    expected = {
        (
            row["dataset"],
            row["model"],
            int(row["seed"]),
            int(row["budget"]),
            row["strategy"],
        ): row
        for row in expected_rows
    }
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise SystemExit(f"trace key mismatch for {task_outcomes_path}: missing={missing} extra={extra}")

    max_error = 0.0
    for key, got in actual.items():
        want = expected[key]
        int_fields = ["completed_tasks", "total_tasks", "is_complete"]
        for field in int_fields:
            if int(float(got[field])) != int(float(want[field])):
                raise SystemExit(f"{field} mismatch for {key}: got {got[field]} want {want[field]}")
        float_fields = [
            "queries",
            "queries_per_task",
            "unsafe_and_functional_at_q",
            "q_at_success",
            "raw_asr",
            "functional_rate",
        ]
        for field in float_fields:
            error = abs(float(got[field]) - float(want[field]))
            max_error = max(max_error, error)
            if error > 1e-9:
                raise SystemExit(f"{field} mismatch for {key}: got {got[field]} want {want[field]}")
    return {
        "task_outcome_rows": len(task_rows),
        "seed_rows_verified": len(actual),
        "max_abs_error": max_error,
    }


def _aggregate_task_outcomes(rows: list[dict[str, str]]) -> dict[tuple[str, str, int, int, str], dict[str, float | int | str]]:
    grouped: dict[tuple[str, str, int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["model"],
            int(row["seed"]),
            int(row["budget"]),
            row["strategy"],
        )
        grouped[key].append(row)

    out: dict[tuple[str, str, int, int, str], dict[str, float | int | str]] = {}
    for key, items in grouped.items():
        completed = len(items)
        queries = sum(float(row["attempts"]) for row in items)
        effective = [float(row["effective_vulnerable"]) for row in items]
        vulnerable = [float(row["vulnerable"]) for row in items]
        functional = [float(row["functional"]) for row in items]
        success_attempts = [
            float(row["attempt_to_success"])
            for row in items
            if float(row["effective_vulnerable"]) > 0 and float(row.get("attempt_to_success", "0") or 0) > 0
        ]
        out[key] = {
            "completed_tasks": completed,
            "total_tasks": completed,
            "is_complete": 1,
            "queries": queries,
            "queries_per_task": queries / max(completed, 1),
            "unsafe_and_functional_at_q": _mean(effective),
            "q_at_success": _mean(success_attempts) if success_attempts else 0.0,
            "raw_asr": _mean(vulnerable),
            "functional_rate": _mean(functional),
        }
    return out


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing trace artifact files: {missing}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    return math.fsum(values) / max(len(values), 1)


if __name__ == "__main__":
    main()
