"""Generate CyberSecEval cost, generalization, and failure diagnostics."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--main-method", default="qscout_qbw_comment")
    args = parser.parse_args()

    root = Path(args.root)
    tables = Path(args.tables)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outcomes = _latest_outcomes(root)
    summary = _read_csv(tables / "ccfa_budget_summary.csv")
    strongest = _read_csv(tables / "ccfa_strongest_baseline_comparison.csv")

    baseline_by_budget = {
        row["budget"]: row["strongest_nonquantum_baseline"]
        for row in strongest
    }
    cost_rows = _cost_rows(summary, strongest, args.main_method)
    language_rows = _slice_rows(outcomes, args.main_method, baseline_by_budget, "language")
    cwe_rows = _slice_rows(outcomes, args.main_method, baseline_by_budget, "cwe")
    failure_rows = _failure_rows(outcomes, args.main_method, baseline_by_budget)

    _write_csv(out_dir / "cyberseceval_cost_efficiency.csv", cost_rows)
    _write_csv(out_dir / "cyberseceval_language_generalization.csv", language_rows)
    _write_csv(out_dir / "cyberseceval_cwe_generalization.csv", cwe_rows)
    _write_csv(out_dir / "cyberseceval_failure_boundary.csv", failure_rows)
    (out_dir / "cyberseceval_diagnostics.md").write_text(
        _markdown(cost_rows, language_rows, cwe_rows, failure_rows),
        encoding="utf-8",
    )
    print(f"diagnostics: {out_dir / 'cyberseceval_diagnostics.md'}")


def _latest_outcomes(root: Path) -> list[dict[str, str]]:
    keyed: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for path in root.glob("*/streaming_task_outcomes.csv"):
        for row in _read_csv(path):
            if row.get("is_complete") == "0":
                continue
            key = (
                row["seed"],
                row["budget"],
                row["strategy"],
                row["task_index"],
                row["task_id"],
            )
            keyed[key] = row
    return list(keyed.values())


def _cost_rows(
    summary: list[dict[str, str]],
    strongest: list[dict[str, str]],
    main_method: str,
) -> list[dict[str, object]]:
    by_key = {
        (row["budget"], row["strategy"]): row
        for row in summary
    }
    out = []
    for row in strongest:
        budget = row["budget"]
        baseline = row["strongest_nonquantum_baseline"]
        main = by_key[(budget, main_method)]
        control = by_key[(budget, baseline)]
        main_qpt = float(main["mean_queries_per_task"])
        control_qpt = float(control["mean_queries_per_task"])
        main_qsuccess = float(main["mean_q_at_success"])
        control_qsuccess = float(control["mean_q_at_success"])
        out.append(
            {
                "budget": budget,
                "main_method": main_method,
                "baseline": baseline,
                "main_unsafe_and_functional": main["mean_unsafe_and_functional_at_q"],
                "baseline_unsafe_and_functional": control["mean_unsafe_and_functional_at_q"],
                "absolute_gain_pp": row["absolute_gain_pp"],
                "main_queries_per_task": f"{main_qpt:.4f}",
                "baseline_queries_per_task": f"{control_qpt:.4f}",
                "queries_per_task_reduction_percent": f"{100.0 * (control_qpt - main_qpt) / max(control_qpt, 1e-12):.2f}",
                "main_q_at_success": f"{main_qsuccess:.4f}",
                "baseline_q_at_success": f"{control_qsuccess:.4f}",
                "q_at_success_reduction_percent": row["q_at_success_reduction_percent"],
            }
        )
    return out


def _slice_rows(
    outcomes: list[dict[str, str]],
    main_method: str,
    baseline_by_budget: dict[str, str],
    field: str,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in outcomes:
        budget = row["budget"]
        baseline = baseline_by_budget.get(budget)
        if row["strategy"] not in {main_method, baseline}:
            continue
        grouped[(budget, field, row[field], row["strategy"])].append(row)

    values = sorted({(budget, value) for budget, _, value, _ in grouped})
    out = []
    for budget, value in values:
        baseline = baseline_by_budget[budget]
        main_items = grouped.get((budget, field, value, main_method), [])
        base_items = grouped.get((budget, field, value, baseline), [])
        if not main_items or not base_items:
            continue
        main_asr = _mean_effective(main_items)
        base_asr = _mean_effective(base_items)
        count = min(len(main_items), len(base_items))
        out.append(
            {
                "budget": budget,
                field: value,
                "tasks_x_seeds": count,
                "main_asr": f"{main_asr:.4f}",
                "baseline": baseline,
                "baseline_asr": f"{base_asr:.4f}",
                "absolute_gain_pp": f"{100.0 * (main_asr - base_asr):+.2f}",
            }
        )
    return out


def _failure_rows(
    outcomes: list[dict[str, str]],
    main_method: str,
    baseline_by_budget: dict[str, str],
) -> list[dict[str, object]]:
    by_key = {
        (row["seed"], row["budget"], row["strategy"], row["task_id"]): row
        for row in outcomes
    }
    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"main_only": 0, "baseline_only": 0, "both": 0, "neither": 0})
    task_keys = {(row["seed"], row["budget"], row["task_id"]) for row in outcomes if row["budget"] in baseline_by_budget}
    for seed, budget, task_id in task_keys:
        baseline = baseline_by_budget[budget]
        main = by_key.get((seed, budget, main_method, task_id))
        base = by_key.get((seed, budget, baseline, task_id))
        if not main or not base:
            continue
        key = (budget, main["cwe"], main["language"])
        main_ok = int(main["effective_vulnerable"])
        base_ok = int(base["effective_vulnerable"])
        if main_ok and base_ok:
            grouped[key]["both"] += 1
        elif main_ok:
            grouped[key]["main_only"] += 1
        elif base_ok:
            grouped[key]["baseline_only"] += 1
        else:
            grouped[key]["neither"] += 1
    rows = []
    for (budget, cwe, language), counts in sorted(grouped.items()):
        total = sum(counts.values())
        if counts["main_only"] == 0 and counts["baseline_only"] == 0:
            continue
        rows.append(
            {
                "budget": budget,
                "cwe": cwe,
                "language": language,
                "main_only": counts["main_only"],
                "baseline_only": counts["baseline_only"],
                "both": counts["both"],
                "neither": counts["neither"],
                "net_main_minus_baseline": counts["main_only"] - counts["baseline_only"],
                "total": total,
            }
        )
    return rows


def _mean_effective(rows: list[dict[str, str]]) -> float:
    return sum(int(row["effective_vulnerable"]) for row in rows) / max(len(rows), 1)


def _markdown(
    cost_rows: list[dict[str, object]],
    language_rows: list[dict[str, object]],
    cwe_rows: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
) -> str:
    lines = ["# CyberSecEval Strong-Baseline Diagnostics", ""]
    lines.append("## Cost Efficiency")
    lines.extend(_table(cost_rows))
    lines.append("")
    lines.append("## Language Generalization")
    lines.extend(_table(language_rows))
    lines.append("")
    lines.append("## CWE Generalization")
    lines.extend(_table(cwe_rows[:30]))
    lines.append("")
    lines.append("## Failure Boundary")
    lines.extend(_table(failure_rows[:30]))
    lines.append("")
    return "\n".join(lines)


def _table(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0])
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return out


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


if __name__ == "__main__":
    main()
