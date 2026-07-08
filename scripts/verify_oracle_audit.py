"""Verify the committed oracle boundary audit."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


AUDIT_ROOT = Path("paper_artifacts/ccfa_oracle_audit_20260708")
TRACE_ROOT = Path("paper_artifacts/ccfa_trace_20260708")
MANIFEST = AUDIT_ROOT / "oracle_audit_manifest.json"
BY_DATASET = AUDIT_ROOT / "oracle_boundary_by_dataset.csv"
BOUNDARY_QUEUE = AUDIT_ROOT / "oracle_boundary_queue.csv"
MAIN_OUTCOMES = TRACE_ROOT / "main_task_outcomes.csv"
CYBER_OUTCOMES = TRACE_ROOT / "cyberseceval_task_outcomes.csv"


def main() -> None:
    report = verify_oracle_audit()
    print("CCF-A oracle audit verification passed.")
    print(json.dumps(report, indent=2))


def verify_oracle_audit() -> dict[str, object]:
    _require_files([MANIFEST, BY_DATASET, BOUNDARY_QUEUE, MAIN_OUTCOMES, CYBER_OUTCOMES])
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    trace_rows = _trace_rows(MAIN_OUTCOMES, "main") + _trace_rows(CYBER_OUTCOMES, "cyberseceval")
    if int(manifest["task_outcome_rows_audited"]) != len(trace_rows):
        raise SystemExit(
            f"oracle audit row mismatch: manifest={manifest['task_outcome_rows_audited']} trace={len(trace_rows)}"
        )
    expected = _aggregate_by_dataset(trace_rows)
    observed = {
        (row["artifact_group"], row["dataset"]): row
        for row in _read_csv(BY_DATASET)
    }
    if set(expected) != set(observed):
        raise SystemExit(f"oracle audit dataset keys mismatch: expected={sorted(expected)} observed={sorted(observed)}")
    max_error = 0.0
    for key, exp in expected.items():
        obs = observed[key]
        for field in ["task_rows", "effective_rows", "vulnerable_rows", "functional_rows"]:
            if int(float(obs[field])) != int(exp[field]):
                raise SystemExit(f"oracle audit {field} mismatch for {key}: {obs[field]} != {exp[field]}")
        for field in ["effective_rate", "vulnerable_nonfunctional_rate", "functional_not_vulnerable_rate"]:
            error = abs(float(obs[field]) - float(exp[field]))
            max_error = max(max_error, error)
            if error > 1e-12:
                raise SystemExit(f"oracle audit {field} mismatch for {key}: {obs[field]} != {exp[field]}")
    boundary_rows = _read_csv(BOUNDARY_QUEUE)
    if not boundary_rows:
        raise SystemExit("oracle boundary queue is empty")
    return {
        "task_outcome_rows": len(trace_rows),
        "dataset_rows_verified": len(expected),
        "boundary_queue_rows": len(boundary_rows),
        "max_abs_error": max_error,
    }


def _trace_rows(path: Path, artifact_group: str) -> list[dict[str, object]]:
    rows = []
    for row in _read_csv(path):
        rows.append(
            {
                "artifact_group": artifact_group,
                "dataset": row["dataset"],
                "vulnerable": int(float(row["vulnerable"])),
                "functional": int(float(row["functional"])),
                "effective": int(float(row["effective_vulnerable"])),
            }
        )
    return rows


def _aggregate_by_dataset(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, float | int]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["artifact_group"]), str(row["dataset"]))].append(row)
    out = {}
    for key, items in grouped.items():
        task_rows = len(items)
        effective = sum(int(row["effective"]) for row in items)
        vulnerable = sum(int(row["vulnerable"]) for row in items)
        functional = sum(int(row["functional"]) for row in items)
        vuln_nonfunc = sum(int(row["vulnerable"]) and not int(row["functional"]) for row in items)
        func_not_vuln = sum(int(row["functional"]) and not int(row["vulnerable"]) for row in items)
        out[key] = {
            "task_rows": task_rows,
            "effective_rows": effective,
            "vulnerable_rows": vulnerable,
            "functional_rows": functional,
            "effective_rate": effective / max(task_rows, 1),
            "vulnerable_nonfunctional_rate": vuln_nonfunc / max(task_rows, 1),
            "functional_not_vulnerable_rate": func_not_vuln / max(task_rows, 1),
        }
    return out


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing oracle audit files: {missing}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
