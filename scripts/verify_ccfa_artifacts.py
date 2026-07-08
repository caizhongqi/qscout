"""Verify the committed CCF-A evidence tables.

This script uses only the Python standard library so reviewers can check table
and compact trace integrity before installing the optional LLM/QPU stacks.

Run:
    python scripts/verify_ccfa_artifacts.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_oracle_audit import verify_oracle_audit
from scripts.verify_trace_artifact import verify_trace_artifact


ROOT = Path("paper_artifacts/ccfa_20260707")
CYBER_ROOT = Path("paper_artifacts/ccfa_20260708")
MAIN_COMPARISON = ROOT / "main_strongest_baseline_comparison.csv"
SEED_ROWS = ROOT / "main_seed_rows.csv"
STRICT_ABLATION = ROOT / "strict_qbw_ablation_budget_summary.csv"
CYBER_COMPARISON = CYBER_ROOT / "ccfa_strongest_baseline_comparison.csv"
CYBER_SEED_ROWS = CYBER_ROOT / "ccfa_seed_rows.csv"
CYBER_SUBSET = CYBER_ROOT / "cyberseceval_autocomplete_subset_120.json"
CYBER_DIAGNOSTICS = [
    CYBER_ROOT / "cyberseceval_cost_efficiency.csv",
    CYBER_ROOT / "cyberseceval_language_generalization.csv",
    CYBER_ROOT / "cyberseceval_cwe_generalization.csv",
    CYBER_ROOT / "cyberseceval_failure_boundary.csv",
    CYBER_ROOT / "cyberseceval_strong_protocol_tables.md",
    CYBER_ROOT / "cyberseceval_strong_diagnostics.md",
]
MECHANISM_FILES = [
    ROOT / "mechanism_securityeval_qwen05.csv",
    ROOT / "mechanism_llmseceval_qwen05.csv",
    ROOT / "mechanism_securityeval_qwen15.csv",
    ROOT / "mechanism_llmseceval_qwen15.csv",
]


def main() -> None:
    _require_files([MAIN_COMPARISON, SEED_ROWS, STRICT_ABLATION, *MECHANISM_FILES])
    _require_files([CYBER_COMPARISON, CYBER_SEED_ROWS, CYBER_SUBSET, *CYBER_DIAGNOSTICS])
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
    if len(seed_rows) < 375:
        raise SystemExit(f"expected at least 375 main seed rows after full-strong baseline expansion, found {len(seed_rows)}")
    seeds = {row["seed"] for row in seed_rows}
    if seeds != {"7", "19", "31", "43", "59"}:
        raise SystemExit(f"expected five seeds, found {sorted(seeds)}")
    _require_strategy_coverage(
        seed_rows,
        dataset="llmseceval",
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        budgets={"4", "8", "16"},
        strategies={
            "fair_random_comment",
            "fair_risk_prior_comment",
            "classical_active_comment",
            "insec_fixed_pool_comment",
            "aot_ensemble_fixed_pool_comment",
            "classical_boundary_witness_comment",
            "qscout_qbw_comment",
        },
    )

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
        if row["strongest_nonquantum_baseline"] != "classical_active_comment":
            raise SystemExit(
                "CyberSecEval strongest baseline should be classical_active_comment "
                f"at budget {row['budget']}, found {row['strongest_nonquantum_baseline']}"
            )
        if float(row["absolute_gain_pp"]) <= 0.0:
            raise SystemExit(f"CyberSecEval non-positive gain at budget {row['budget']}")
        if float(row["paired_delta_ci_low_pp"]) <= 0.0:
            raise SystemExit(f"CyberSecEval CI lower bound is not positive at budget {row['budget']}")

    cyber_seed_rows = _read_csv(CYBER_SEED_ROWS)
    cyber_seeds = {row["seed"] for row in cyber_seed_rows}
    if cyber_seeds != {"7", "19", "31", "43", "59"}:
        raise SystemExit(f"expected five CyberSecEval seeds, found {sorted(cyber_seeds)}")
    cyber_settings = {(row["budget"], row["strategy"]) for row in cyber_seed_rows}
    expected_cyber_strategies = {
        "fair_random_comment",
        "classical_active_comment",
        "insec_fixed_pool_comment",
        "aot_ensemble_fixed_pool_comment",
        "classical_boundary_witness_comment",
        "qscout_qbw_comment",
    }
    for budget in {"4", "8"}:
        strategies = {strategy for row_budget, strategy in cyber_settings if row_budget == budget}
        if strategies != expected_cyber_strategies:
            raise SystemExit(f"unexpected CyberSecEval strategies at B={budget}: {sorted(strategies)}")

    trace_report = verify_trace_artifact()
    oracle_report = verify_oracle_audit()
    print("CCF-A artifact verification passed.")
    print(
        f"main_rows={len(main_rows)} seed_rows={len(seed_rows)} "
        f"mechanism_files={len(MECHANISM_FILES)} cyber_rows={len(cyber_rows)} "
        f"trace_main_rows={trace_report['main']['task_outcome_rows']} "
        f"trace_cyber_rows={trace_report['cyberseceval']['task_outcome_rows']} "
        f"oracle_rows={oracle_report['task_outcome_rows']}"
    )


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing artifact files: {missing}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require_strategy_coverage(
    rows: list[dict[str, str]],
    *,
    dataset: str,
    model: str,
    budgets: set[str],
    strategies: set[str],
) -> None:
    for budget in budgets:
        observed = {
            row["strategy"]
            for row in rows
            if row["dataset"] == dataset and row["model"] == model and row["budget"] == budget
        }
        if observed != strategies:
            raise SystemExit(
                f"unexpected strategy coverage for {dataset}/{model}/B={budget}: {sorted(observed)}"
            )


if __name__ == "__main__":
    main()
