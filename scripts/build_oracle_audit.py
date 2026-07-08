"""Build a trace-level oracle boundary audit.

This script does not read or commit generated-code payloads.  It audits the
detector outcomes already committed in the compact trace ledger and separates
effective unsafe/functioning successes from boundary cases such as vulnerable
but non-functional completions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


TRACE_ROOT = Path("paper_artifacts/ccfa_trace_20260708")
DEFAULT_MAIN_OUTCOMES = TRACE_ROOT / "main_task_outcomes.csv"
DEFAULT_CYBER_OUTCOMES = TRACE_ROOT / "cyberseceval_task_outcomes.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-outcomes", default=str(DEFAULT_MAIN_OUTCOMES))
    parser.add_argument("--cyber-outcomes", default=str(DEFAULT_CYBER_OUTCOMES))
    parser.add_argument("--output-dir", default="paper_artifacts/ccfa_oracle_audit_20260708")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_outcomes(Path(args.main_outcomes), "main") + _load_outcomes(Path(args.cyber_outcomes), "cyberseceval")

    summary = _aggregate(rows, ("artifact_group", "dataset", "model", "budget", "strategy"))
    by_dataset = _aggregate(rows, ("artifact_group", "dataset"))
    by_cwe = _aggregate(rows, ("artifact_group", "dataset", "cwe"))
    by_language = _aggregate(rows, ("artifact_group", "dataset", "language"))
    boundary_queue = _boundary_queue(rows)

    _write_csv(out_dir / "oracle_boundary_summary.csv", summary)
    _write_csv(out_dir / "oracle_boundary_by_dataset.csv", by_dataset)
    _write_csv(out_dir / "oracle_boundary_by_cwe.csv", by_cwe)
    _write_csv(out_dir / "oracle_boundary_by_language.csv", by_language)
    _write_csv(out_dir / "oracle_boundary_queue.csv", boundary_queue)

    manifest = {
        "task_outcome_rows_audited": len(rows),
        "summary_rows": len(summary),
        "cwe_rows": len(by_cwe),
        "language_rows": len(by_language),
        "boundary_queue_rows": len(boundary_queue),
        "policy": "trace-level detector boundary audit; no generated-code payload is committed",
    }
    (out_dir / "oracle_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "ORACLE_AUDIT.md").write_text(_markdown(manifest, by_dataset), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _load_outcomes(path: Path, artifact_group: str) -> list[dict[str, object]]:
    out = []
    for row in _read_csv(path):
        vulnerable = int(float(row["vulnerable"]))
        functional = int(float(row["functional"]))
        effective = int(float(row["effective_vulnerable"]))
        attempts = int(float(row["attempts"]))
        attempt_to_success = int(float(row.get("attempt_to_success", "0") or 0))
        if effective:
            boundary_class = "effective_unsafe_functional"
        elif vulnerable and not functional:
            boundary_class = "vulnerable_nonfunctional"
        elif functional and not vulnerable:
            boundary_class = "functional_not_vulnerable"
        elif vulnerable and functional and not effective:
            boundary_class = "inconsistent_vulnerable_functional"
        else:
            boundary_class = "neither_vulnerable_nor_functional"
        item: dict[str, object] = {
            "artifact_group": artifact_group,
            "source_root": row["source_root"],
            "dataset": row["dataset"],
            "model": row["model"],
            "seed": int(float(row["seed"])),
            "budget": int(float(row["budget"])),
            "strategy": row["strategy"],
            "task_index": int(float(row["task_index"])),
            "task_id": row["task_id"],
            "language": row.get("language", ""),
            "cwe": row.get("cwe", ""),
            "attempts": attempts,
            "candidate_pool_size": int(float(row.get("candidate_pool_size", "0") or 0)),
            "vulnerable": vulnerable,
            "functional": functional,
            "effective_vulnerable": effective,
            "attempt_to_success": attempt_to_success,
            "boundary_class": boundary_class,
            "task_key_sha256": row.get("task_key_sha256", ""),
        }
        out.append(item)
    return out


def _aggregate(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    out: list[dict[str, object]] = []
    for key, items in sorted(grouped.items()):
        task_rows = len(items)
        effective = sum(int(row["effective_vulnerable"]) for row in items)
        vulnerable = sum(int(row["vulnerable"]) for row in items)
        functional = sum(int(row["functional"]) for row in items)
        vuln_nonfunc = sum(int(row["vulnerable"]) and not int(row["functional"]) for row in items)
        func_not_vuln = sum(int(row["functional"]) and not int(row["vulnerable"]) for row in items)
        neither = sum((not int(row["functional"])) and (not int(row["vulnerable"])) for row in items)
        success_attempts = [int(row["attempt_to_success"]) for row in items if int(row["attempt_to_success"]) > 0]
        attempts = [int(row["attempts"]) for row in items]
        first_attempt_successes = sum(int(row["attempt_to_success"]) == 1 for row in items)
        row_out = {keys[index]: key[index] for index in range(len(keys))}
        row_out.update(
            {
                "task_rows": task_rows,
                "effective_rows": effective,
                "vulnerable_rows": vulnerable,
                "functional_rows": functional,
                "effective_rate": _rate(effective, task_rows),
                "vulnerable_nonfunctional_rate": _rate(vuln_nonfunc, task_rows),
                "functional_not_vulnerable_rate": _rate(func_not_vuln, task_rows),
                "neither_rate": _rate(neither, task_rows),
                "mean_attempts": _mean(attempts),
                "mean_attempt_to_success": _mean(success_attempts),
                "first_attempt_success_rate": _rate(first_attempt_successes, task_rows),
            }
        )
        out.append(row_out)
    return out


def _boundary_queue(rows: list[dict[str, object]], limit: int = 500) -> list[dict[str, object]]:
    priority = {
        "vulnerable_nonfunctional": 0,
        "inconsistent_vulnerable_functional": 1,
        "functional_not_vulnerable": 2,
        "neither_vulnerable_nor_functional": 3,
        "effective_unsafe_functional": 4,
    }
    flagged = [row for row in rows if row["boundary_class"] != "effective_unsafe_functional"]
    flagged.sort(
        key=lambda row: (
            priority.get(str(row["boundary_class"]), 99),
            str(row["dataset"]),
            str(row["model"]),
            int(row["budget"]),
            str(row["strategy"]),
            int(row["seed"]),
            int(row["task_index"]),
        )
    )
    keep = []
    for row in flagged[:limit]:
        keep.append(
            {
                "artifact_group": row["artifact_group"],
                "dataset": row["dataset"],
                "model": row["model"],
                "seed": row["seed"],
                "budget": row["budget"],
                "strategy": row["strategy"],
                "task_id": row["task_id"],
                "language": row["language"],
                "cwe": row["cwe"],
                "attempts": row["attempts"],
                "boundary_class": row["boundary_class"],
                "task_key_sha256": row["task_key_sha256"],
            }
        )
    return keep


def _markdown(manifest: dict[str, object], by_dataset: list[dict[str, object]]) -> str:
    lines = [
        "# QScout CCF-A Oracle Boundary Audit",
        "",
        "This audit checks the committed detector/task-outcome ledger.  It does not claim",
        "human-level exploit validation and does not commit generated-code payloads.",
        "",
        f"- Task outcome rows audited: {manifest['task_outcome_rows_audited']}",
        f"- Boundary queue rows retained: {manifest['boundary_queue_rows']}",
        "",
        "## Dataset Summary",
        "",
        "| Artifact | Dataset | Task rows | Effective rate | Vulnerable nonfunctional | Functional not vulnerable | Mean attempts |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in by_dataset:
        lines.append(
            f"| {row['artifact_group']} | {row['dataset']} | {row['task_rows']} | "
            f"{float(row['effective_rate']):.4f} | "
            f"{float(row['vulnerable_nonfunctional_rate']):.4f} | "
            f"{float(row['functional_not_vulnerable_rate']):.4f} | "
            f"{float(row['mean_attempts']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `effective_rate` is the committed Unsafe-and-Functional task outcome rate at the task-at-budget level.",
            "- `vulnerable_nonfunctional_rate` estimates how often the vulnerability detector fires on code that the functionality heuristic does not accept.",
            "- `functional_not_vulnerable_rate` estimates benign or safe functional completions.",
            "- `oracle_boundary_queue.csv` is a hash-only queue for human/static/unit-test follow-up; it is not removed from the main result.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def _mean(values: list[int]) -> float:
    return sum(values) / max(len(values), 1)


if __name__ == "__main__":
    main()
