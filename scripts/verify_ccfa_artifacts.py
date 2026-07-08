"""Verify the committed CCF-A evidence tables.

This script is intentionally lightweight: it uses only the Python standard
library so reviewers can check table integrity before installing the optional
LLM/QPU stacks.

Run:
    python scripts/verify_ccfa_artifacts.py
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("paper_artifacts/ccfa_20260707")
CYBER_ROOT = Path("paper_artifacts/ccfa_20260708")
MAIN_COMPARISON = ROOT / "main_strongest_baseline_comparison.csv"
SEED_ROWS = ROOT / "main_seed_rows.csv"
STRICT_ABLATION = ROOT / "strict_qbw_ablation_budget_summary.csv"
CYBER_COMPARISON = CYBER_ROOT / "ccfa_strongest_baseline_comparison.csv"
CYBER_SEED_ROWS = CYBER_ROOT / "ccfa_seed_rows.csv"
CYBER_SUBSET = CYBER_ROOT / "cyberseceval_autocomplete_subset_120.json"
MECHANISM_FILES = [
    ROOT / "mechanism_securityeval_qwen05.csv",
    ROOT / "mechanism_llmseceval_qwen05.csv",
    ROOT / "mechanism_securityeval_qwen15.csv",
    ROOT / "mechanism_llmseceval_qwen15.csv",
]


def main() -> None:
    _require_files([MAIN_COMPARISON, SEED_ROWS, STRICT_ABLATION, *MECHANISM_FILES])
    _require_files([CYBER_COMPARISON, CYBER_SEED_ROWS, CYBER_SUBSET])
    main_rows = _read_csv(MAIN_COMPARISON)
    if len(main_rows) != 12:
        raise SystemExit(f"expected 12 main comparison rows, found {len(main_rows)}")
    failed = [
        row
        for row in main_rows
        if float(row["main_unsafe_and_functional_at_q"]) < float(row["baseline_unsafe_and_functional_at_q"])
    ]
    if failed:
        raise SystemExit(f"QScout underperformed baseline in {len(failed)} main rows")

    settings = {(row["dataset"], row["model"]) for row in main_rows}
    expected_settings = {
        ("securityeval", "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
        ("llmseceval", "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
        ("securityeval", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        ("llmseceval", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
    }
    if settings != expected_settings:
        raise SystemExit(f"unexpected setting coverage: {sorted(settings)}")

    seed_rows = _read_csv(SEED_ROWS)
    seeds = {row["seed"] for row in seed_rows}
    if seeds != {"7", "19", "31", "43", "59"}:
        raise SystemExit(f"expected five seeds, found {sorted(seeds)}")

    for path in MECHANISM_FILES:
        rows = _read_csv(path)
        by_signal = {row["signal"]: row for row in rows}
        if "actual_qbw_acquisition" not in by_signal or "qbw_score" not in by_signal:
            raise SystemExit(f"missing mechanism signals in {path}")
        actual_auc = float(by_signal["actual_qbw_acquisition"]["auc"])
        raw_auc = float(by_signal["qbw_score"]["auc"])
        if actual_auc <= raw_auc:
            raise SystemExit(f"objective-aligned acquisition does not improve raw QBW AUC in {path}")

    strict_rows = _read_csv(STRICT_ABLATION)
    strict_strategies = {row["strategy"] for row in strict_rows}
    required_strict = {
        "classical_boundary_witness_comment",
        "qbw_strict_full_comment",
        "qbw_strict_no_density_matrix_comment",
        "qbw_strict_no_fidelity_margin_comment",
        "qbw_strict_no_reliability_penalty_comment",
        "qbw_strict_born_entropy_only_comment",
        "qbw_strict_random_quantum_score_comment",
    }
    if not required_strict.issubset(strict_strategies):
        missing = sorted(required_strict - strict_strategies)
        raise SystemExit(f"missing strict ablation strategies: {missing}")

    cyber_rows = _read_csv(CYBER_COMPARISON)
    cyber_budgets = {row["budget"] for row in cyber_rows}
    if cyber_budgets != {"4", "8"}:
        raise SystemExit(f"unexpected CyberSecEval budget coverage: {sorted(cyber_budgets)}")
    for row in cyber_rows:
        if row["dataset"] != "cyberseceval":
            raise SystemExit(f"unexpected CyberSecEval dataset label: {row['dataset']}")
        if float(row["absolute_gain_pp"]) <= 0.0:
            raise SystemExit(f"CyberSecEval non-positive gain at budget {row['budget']}")

    cyber_seed_rows = _read_csv(CYBER_SEED_ROWS)
    cyber_seeds = {row["seed"] for row in cyber_seed_rows}
    if cyber_seeds != {"7", "19", "31", "43", "59"}:
        raise SystemExit(f"expected five CyberSecEval seeds, found {sorted(cyber_seeds)}")
    cyber_settings = {(row["budget"], row["strategy"]) for row in cyber_seed_rows}
    required_cyber_settings = {
        ("4", "classical_boundary_witness_comment"),
        ("4", "qscout_qbw_comment"),
        ("8", "classical_boundary_witness_comment"),
        ("8", "qscout_qbw_comment"),
    }
    if cyber_settings != required_cyber_settings:
        raise SystemExit(f"unexpected CyberSecEval setting coverage: {sorted(cyber_settings)}")

    print("CCF-A artifact verification passed.")
    print(
        f"main_rows={len(main_rows)} seed_rows={len(seed_rows)} "
        f"mechanism_files={len(MECHANISM_FILES)} cyber_rows={len(cyber_rows)}"
    )


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing artifact files: {missing}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
