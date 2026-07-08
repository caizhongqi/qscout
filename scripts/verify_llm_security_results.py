"""Verify the reviewer-facing `results/llm_security` mirror."""

from __future__ import annotations

import csv
import json
from pathlib import Path


RESULTS_ROOT = Path("results/llm_security")
SUMMARY = RESULTS_ROOT / "summary.csv"
STRONGEST = RESULTS_ROOT / "strongest_baseline_comparison.csv"
MANIFEST = RESULTS_ROOT / "RESULTS_MANIFEST.json"
TRACE_DIR = RESULTS_ROOT / "traces"

MAIN_SUMMARY = Path("paper_artifacts/ccfa_20260707/main_budget_summary.csv")
CYBER_SUMMARY = Path("paper_artifacts/ccfa_20260708/ccfa_budget_summary.csv")
MAIN_STRONGEST = Path("paper_artifacts/ccfa_20260707/main_strongest_baseline_comparison.csv")
CYBER_STRONGEST = Path("paper_artifacts/ccfa_20260708/ccfa_strongest_baseline_comparison.csv")


def main() -> None:
    report = verify_llm_security_results()
    print("LLM security results mirror verification passed.")
    print(json.dumps(report, indent=2))


def verify_llm_security_results() -> dict[str, object]:
    _require_files([SUMMARY, STRONGEST, MANIFEST, MAIN_SUMMARY, CYBER_SUMMARY, MAIN_STRONGEST, CYBER_STRONGEST])
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    summary_rows = _read_csv(SUMMARY)
    expected_budget_rows = _read_csv(MAIN_SUMMARY) + _read_csv(CYBER_SUMMARY)
    strongest_rows = _read_csv(STRONGEST)
    expected_strongest = _read_csv(MAIN_STRONGEST) + _read_csv(CYBER_STRONGEST)
    if int(manifest["summary_rows"]) != len(summary_rows):
        raise SystemExit("summary row count does not match manifest")
    if len(summary_rows) != len(expected_budget_rows):
        raise SystemExit(f"summary rows {len(summary_rows)} != budget artifact rows {len(expected_budget_rows)}")
    if len(strongest_rows) != len(expected_strongest):
        raise SystemExit(f"strongest rows {len(strongest_rows)} != artifact rows {len(expected_strongest)}")

    trace_counts: dict[str, int] = {}
    for path in sorted(TRACE_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            trace_counts[path.name] = sum(1 for _ in handle)
    if trace_counts != dict(manifest["trace_jsonl_files"]):
        raise SystemExit(f"trace counts mismatch: {trace_counts} != {manifest['trace_jsonl_files']}")
    return {
        "summary_rows": len(summary_rows),
        "strongest_rows": len(strongest_rows),
        "trace_counts": trace_counts,
    }


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing results files: {missing}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
