"""Generate CCF-A style fair-pool protocol tables from streaming runs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


QUANTUM_MARKERS = (
    "qbw",
    "qscout",
    "helstrom",
    "grover",
    "physics",
    "calibrated_qql",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", required=True, help="Comma-separated streaming output roots.")
    parser.add_argument("--output-dir", default="outputs/ccfa_protocol_tables_20260706")
    parser.add_argument("--main-method", default="qscout_qbw_comment")
    args = parser.parse_args()

    roots = [Path(item.strip()) for item in args.roots.split(",") if item.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = _load_seed_rows(roots)
    summary_rows = _aggregate_by_budget(seed_rows)
    aulc_rows = _aulc_rows(summary_rows)
    baseline_rows = _strongest_baseline_rows(summary_rows, args.main_method, seed_rows)
    gate_rows = _gate_rows(summary_rows, aulc_rows, args.main_method, seed_rows)

    _write_csv(out_dir / "ccfa_seed_rows.csv", seed_rows)
    _write_csv(out_dir / "ccfa_budget_summary.csv", summary_rows)
    _write_csv(out_dir / "ccfa_aulc_summary.csv", aulc_rows)
    _write_csv(out_dir / "ccfa_strongest_baseline_comparison.csv", baseline_rows)
    _write_csv(out_dir / "ccfa_gate_summary.csv", gate_rows)
    (out_dir / "ccfa_protocol_tables.md").write_text(
        _markdown(summary_rows, aulc_rows, baseline_rows, gate_rows, args.main_method),
        encoding="utf-8",
    )
    print(f"tables: {out_dir / 'ccfa_protocol_tables.md'}")


def _load_seed_rows(roots: list[Path]) -> list[dict[str, object]]:
    out = []
    for root in roots:
        for path in root.glob("*/streaming_summary.csv"):
            for row in _read_csv(path):
                unsafe = row.get("unsafe_and_functional_at_q") or row.get("effective_asr_completed") or "0"
                q_success = row.get("q_at_success_completed") or "0"
                out.append(
                    {
                        "source_root": str(root),
                        "dataset": row["dataset"],
                        "model": row["model"],
                        "seed": int(row["seed"]),
                        "budget": int(row["budget"]),
                        "strategy": row["strategy"],
                        "completed_tasks": int(row["completed_tasks"]),
                        "total_tasks": int(row["total_tasks"]),
                        "is_complete": int(row["is_complete"]),
                        "queries": float(row["queries"]),
                        "queries_per_task": float(row["queries_per_completed_task"]),
                        "unsafe_and_functional_at_q": float(unsafe),
                        "q_at_success": float(q_success),
                        "raw_asr": float(row.get("raw_asr_completed", "0") or 0),
                        "functional_rate": float(row.get("functional_rate_completed", "0") or 0),
                    }
                )
    return _dedupe_seed_rows(out)


def _dedupe_seed_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keyed: dict[tuple[str, str, int, int, str], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["model"]),
            int(row["seed"]),
            int(row["budget"]),
            str(row["strategy"]),
        )
        current = keyed.get(key)
        if current is None or _prefer_seed_row(row, current):
            keyed[key] = row
    return sorted(
        keyed.values(),
        key=lambda row: (
            str(row["dataset"]),
            str(row["model"]),
            int(row["budget"]),
            str(row["strategy"]),
            int(row["seed"]),
        ),
    )


def _prefer_seed_row(candidate: dict[str, object], current: dict[str, object]) -> bool:
    candidate_complete = int(candidate.get("is_complete", 0))
    current_complete = int(current.get("is_complete", 0))
    if candidate_complete != current_complete:
        return candidate_complete > current_complete
    candidate_tasks = int(candidate.get("completed_tasks", 0))
    current_tasks = int(current.get("completed_tasks", 0))
    if candidate_tasks != current_tasks:
        return candidate_tasks > current_tasks
    candidate_source = str(candidate.get("source_root", "")).replace("\\", "/")
    current_source = str(current.get("source_root", "")).replace("\\", "/")
    return "/main" in candidate_source and "/main" not in current_source


def _aggregate_by_budget(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["is_complete"]) != 1:
            continue
        grouped[(str(row["dataset"]), str(row["model"]), int(row["budget"]), str(row["strategy"]))].append(row)
    out = []
    for (dataset, model, budget, strategy), items in sorted(grouped.items()):
        unsafe = [float(row["unsafe_and_functional_at_q"]) for row in items]
        q_success = [float(row["q_at_success"]) for row in items if float(row["q_at_success"]) > 0]
        queries = [float(row["queries"]) for row in items]
        qpt = [float(row["queries_per_task"]) for row in items]
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "budget": budget,
                "strategy": strategy,
                "display_strategy": _display_strategy(strategy),
                "seeds": ",".join(str(row["seed"]) for row in sorted(items, key=lambda item: int(item["seed"]))),
                "mean_unsafe_and_functional_at_q": statistics.fmean(unsafe),
                "ci95_unsafe_and_functional_at_q": _ci95(unsafe),
                "mean_q_at_success": statistics.fmean(q_success) if q_success else 0.0,
                "mean_queries": statistics.fmean(queries),
                "mean_queries_per_task": statistics.fmean(qpt),
                "complete_seed_count": len(items),
                "min_completed_tasks": min(int(row["completed_tasks"]) for row in items),
                "max_completed_tasks": max(int(row["completed_tasks"]) for row in items),
            }
        )
    return out


def _aulc_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(str(row["dataset"]), str(row["model"]), str(row["strategy"]))].append(row)
    out = []
    for (dataset, model, strategy), items in sorted(grouped.items()):
        points = sorted((int(row["budget"]), float(row["mean_unsafe_and_functional_at_q"])) for row in items)
        if len(points) == 1:
            aulc = points[0][1]
        else:
            area = 0.0
            for (x0, y0), (x1, y1) in zip(points, points[1:]):
                area += (x1 - x0) * (y0 + y1) / 2.0
            aulc = area / max(points[-1][0] - points[0][0], 1)
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "strategy": strategy,
                "display_strategy": _display_strategy(strategy),
                "budgets": ",".join(str(point[0]) for point in points),
                "aulc": aulc,
                "min_budget": points[0][0],
                "max_budget": points[-1][0],
            }
        )
    return out


def _strongest_baseline_rows(
    summary_rows: list[dict[str, object]],
    main_method: str,
    seed_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(str(row["dataset"]), str(row["model"]), int(row["budget"]))].append(row)
    out = []
    for (dataset, model, budget), items in sorted(grouped.items()):
        main = next((row for row in items if row["strategy"] == main_method), None)
        if main is None and main_method == "qscout_qbw_comment":
            main = next((row for row in items if row["strategy"] == "qbw_qql_comment"), None)
        if main is None:
            continue
        baselines = [row for row in items if not _is_quantum_strategy(str(row["strategy"]))]
        if not baselines:
            continue
        best = max(baselines, key=lambda row: float(row["mean_unsafe_and_functional_at_q"]))
        main_unsafe = float(main["mean_unsafe_and_functional_at_q"])
        base_unsafe = float(best["mean_unsafe_and_functional_at_q"])
        main_q = float(main["mean_q_at_success"])
        base_q = float(best["mean_q_at_success"])
        paired = _paired_seed_deltas(
            seed_rows or [],
            dataset,
            model,
            budget,
            str(main["strategy"]),
            str(best["strategy"]),
        )
        delta_ci = _ci95(paired)
        delta_mean = statistics.fmean(paired) if paired else main_unsafe - base_unsafe
        delta_low = delta_mean - delta_ci
        delta_high = delta_mean + delta_ci
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "budget": budget,
                "main_method": str(main["strategy"]),
                "strongest_nonquantum_baseline": str(best["strategy"]),
                "main_unsafe_and_functional_at_q": main_unsafe,
                "baseline_unsafe_and_functional_at_q": base_unsafe,
                "absolute_gain_pp": (main_unsafe - base_unsafe) * 100.0,
                "relative_gain_percent": (main_unsafe - base_unsafe) / max(base_unsafe, 1e-12) * 100.0,
                "paired_delta_ci95_pp": delta_ci * 100.0,
                "paired_delta_ci_low_pp": delta_low * 100.0,
                "paired_delta_ci_high_pp": delta_high * 100.0,
                "paired_delta_ci_excludes_zero": int(delta_low > 0.0 or delta_high < 0.0),
                "main_q_at_success": main_q,
                "baseline_q_at_success": base_q,
                "q_at_success_reduction_percent": (base_q - main_q) / max(base_q, 1e-12) * 100.0 if main_q > 0 and base_q > 0 else 0.0,
            }
        )
    return out


def _gate_rows(
    summary_rows: list[dict[str, object]],
    aulc_rows: list[dict[str, object]],
    main_method: str,
    seed_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    comparisons = _strongest_baseline_rows(summary_rows, main_method, seed_rows)
    aulc_lookup = {
        (row["dataset"], row["model"], row["strategy"]): float(row["aulc"])
        for row in aulc_rows
    }
    baseline_by_setting: dict[tuple[str, str], str] = {}
    for row in comparisons:
        key = (str(row["dataset"]), str(row["model"]))
        baseline_by_setting[key] = str(row["strongest_nonquantum_baseline"])
    out = []
    for row in comparisons:
        key = (str(row["dataset"]), str(row["model"]))
        main_strategy = str(row["main_method"])
        base_strategy = baseline_by_setting.get(key, str(row["strongest_nonquantum_baseline"]))
        main_aulc = aulc_lookup.get((key[0], key[1], main_strategy), 0.0)
        base_aulc = aulc_lookup.get((key[0], key[1], base_strategy), 0.0)
        aulc_gain = (main_aulc - base_aulc) / max(base_aulc, 1e-12) * 100.0 if base_aulc > 0 else 0.0
        pass_gate = (
            float(row["absolute_gain_pp"]) >= 5.0
            or float(row["q_at_success_reduction_percent"]) >= 20.0
            or aulc_gain >= 10.0
        )
        strict_pass = pass_gate and int(row.get("paired_delta_ci_excludes_zero", 0)) == 1
        out.append(
            {
                **row,
                "main_aulc": main_aulc,
                "baseline_aulc": base_aulc,
                "aulc_gain_percent": aulc_gain,
                "passes_ccfa_effect_gate": int(pass_gate),
                "passes_strict_ci_gate": int(strict_pass),
            }
        )
    return out


def _markdown(
    summary_rows: list[dict[str, object]],
    aulc_rows: list[dict[str, object]],
    baseline_rows: list[dict[str, object]],
    gate_rows: list[dict[str, object]],
    main_method: str,
) -> str:
    lines = [
        "# CCF-A Fair-Pool Protocol Tables",
        "",
        "## Protocol",
        "",
        "- All fixed-pool baselines rank or sample from the same candidate prompt/code mutation pool.",
        "- Main metric: Unsafe-and-Functional@Q.",
        "- Main comparison: QScout-QBW vs the strongest non-quantum baseline available in the same setting.",
        "- Gate: +5 pp Unsafe-and-Functional@Q, or >=20% Q@Success reduction, or >=10% AULC gain.",
        "",
        "## Budget Summary",
        "",
        "| Dataset | Model | Budget | Method | Unsafe-and-Functional@Q | 95% CI | Q@Success | Queries/task | Seeds |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['budget']} | {row['display_strategy']} | {float(row['mean_unsafe_and_functional_at_q']):.4f} | +/- {float(row['ci95_unsafe_and_functional_at_q']):.4f} | {float(row['mean_q_at_success']):.2f} | {float(row['mean_queries_per_task']):.2f} | {row['seeds']} |"
        )
    lines.extend(["", "## AULC", "", "| Dataset | Model | Method | Budgets | AULC |", "|---|---|---|---|---:|"])
    for row in aulc_rows:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['display_strategy']} | {row['budgets']} | {float(row['aulc']):.4f} |")
    lines.extend(["", "## Strongest Non-Quantum Baseline Comparison", "", "| Dataset | Model | Budget | Main | Strongest baseline | Abs. gain | Paired 95% CI | Rel. gain | Q@Success reduction |", "|---|---|---:|---|---|---:|---:|---:|---:|"])
    for row in baseline_rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['budget']} | {_display_strategy(str(row['main_method']))} | {_display_strategy(str(row['strongest_nonquantum_baseline']))} | {float(row['absolute_gain_pp']):+.2f} pp | [{float(row['paired_delta_ci_low_pp']):+.2f}, {float(row['paired_delta_ci_high_pp']):+.2f}] | {float(row['relative_gain_percent']):+.2f}% | {float(row['q_at_success_reduction_percent']):+.2f}% |"
        )
    lines.extend(["", "## Gate Summary", "", "| Dataset | Model | Budget | Effect gate | Strict CI gate | AULC gain |", "|---|---|---:|---:|---:|---:|"])
    for row in gate_rows:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['budget']} | {row['passes_ccfa_effect_gate']} | {row['passes_strict_ci_gate']} | {float(row['aulc_gain_percent']):+.2f}% |")
    return "\n".join(lines) + "\n"


def _is_quantum_strategy(strategy: str) -> bool:
    lower = strategy.lower()
    return any(marker in lower for marker in QUANTUM_MARKERS)


def _display_strategy(strategy: str) -> str:
    return {
        "fair_random_comment": "Random Search",
        "fair_risk_prior_comment": "Risk Prior",
        "classical_active_comment": "Classical Active",
        "insec_fixed_pool_comment": "INSEC-style Fixed-Pool Search",
        "aot_ensemble_fixed_pool_comment": "AOT-style Ensemble",
        "classical_boundary_witness_comment": "Classical Boundary Witness",
        "qfrontier_qsfa_comment": "QSFA",
        "qbw_qql_comment": "QScout-QBW",
        "qscout_qbw_comment": "QScout-QBW",
        "helstrom_qbw_qql_comment": "Helstrom-QBW",
        "qbw_no_density_matrix_comment": "QScout-QBW w/o density matrix",
        "qbw_no_fidelity_margin_comment": "QScout-QBW w/o fidelity margin",
        "qbw_no_reliability_penalty_comment": "QScout-QBW w/o reliability penalty",
        "qbw_born_entropy_only_comment": "Born entropy only",
        "qbw_random_quantum_score_comment": "Random quantum score",
    }.get(strategy, strategy)


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _paired_seed_deltas(
    seed_rows: list[dict[str, object]],
    dataset: str,
    model: str,
    budget: int,
    main_strategy: str,
    baseline_strategy: str,
) -> list[float]:
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for row in seed_rows:
        if (
            str(row["dataset"]) != str(dataset)
            or str(row["model"]) != str(model)
            or int(row["budget"]) != int(budget)
            or int(row.get("is_complete", 0)) != 1
        ):
            continue
        by_seed[int(row["seed"])][str(row["strategy"])] = float(row["unsafe_and_functional_at_q"])
    out = []
    for values in by_seed.values():
        if main_strategy in values and baseline_strategy in values:
            out.append(values[main_strategy] - values[baseline_strategy])
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
