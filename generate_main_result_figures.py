"""Generate main-paper figures only from completed multi-seed QScout studies.

Expected inputs are outputs/study_*.csv produced by run_study_matrix.py.
The script refuses to turn a single-seed pilot into a main result. It also
exports every failed or excluded cell to an appendix table.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
        return value if math.isfinite(value) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def clone_metric(row: dict[str, str]) -> float:
    if row.get("strategy") == "active":
        value = number(row, "hybrid_clone_agreement")
        if math.isfinite(value):
            return value
    if row.get("strategy") == "classical_active":
        value = number(row, "classical_active_clone_agreement")
        if math.isfinite(value):
            return value
    return number(row, "mlp_agreement")


def collect_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(OUT.glob("study_*.csv")):
        if path.name.endswith("_summary.csv") or path.name.endswith("_paired_policy_comparison.csv"):
            continue
        loaded = read_csv(path)
        for row in loaded:
            row["source_file"] = path.name
        rows.extend(loaded)
    return rows


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=260, bbox_inches="tight")
    fig.savefig(FIG / name.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)


def require_multiseed(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    return {key: value for key, value in groups.items() if len({row.get("seed") for row in value}) >= 2}


def fidelity_query(rows: list[dict[str, str]], report: list[str]) -> None:
    keys = ("dataset", "victim", "strategy", "budget", "qubits", "layers", "noise_kind", "noise_p", "defense_label_noise")
    groups = require_multiseed(rows, keys)
    values: dict[tuple[str, str, str], list[tuple[int, float, float]]] = defaultdict(list)
    for key, group in groups.items():
        score = np.asarray([clone_metric(row) for row in group], dtype=float)
        score = score[np.isfinite(score)]
        if not len(score):
            continue
        dataset, victim, strategy, budget, *_ = key
        values[(dataset, victim, strategy)].append((int(budget), float(score.mean()), float(score.std(ddof=1) / np.sqrt(len(score)))))
    if not values:
        report.append("fig10 F-Q skipped: no multi-seed study rows.")
        return
    panels = sorted({(dataset, victim) for dataset, victim, _ in values})
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.2), squeeze=False)
    colors = {"random": "#8d5a35", "classical_active": "#3b6f9c", "active": "#2f6b5f"}
    labels = {"random": "Random", "classical_active": "Classical-active", "active": "Q-Scout guided"}
    for ax, panel in zip(axes[0], panels):
        for (dataset, victim, strategy), points in values.items():
            if (dataset, victim) != panel:
                continue
            points.sort()
            x, y, err = map(np.asarray, zip(*points))
            ax.plot(x, y, marker="o", lw=2, label=labels.get(strategy, strategy), color=colors.get(strategy, "#555555"))
            ax.fill_between(x, y - 1.96 * err, y + 1.96 * err, alpha=0.14, color=colors.get(strategy, "#555555"))
        ax.set_title(f"{panel[0]} / {panel[1]}")
        ax.set_xlabel("Hard-label query budget")
        ax.set_ylabel("Clone fidelity")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0][0].legend(frameon=False)
    save(fig, "fig10_fidelity_query_curves.png")
    report.append("fig10 F-Q generated from multi-seed clone fidelity.")


def paired_forest(report: list[str]) -> None:
    rows: list[dict[str, str]] = []
    for path in sorted(OUT.glob("study_*_paired_policy_comparison.csv")):
        rows.extend(read_csv(path))
    if not rows:
        report.append("fig11 forest skipped: no paired policy CSV.")
        return
    rows.sort(key=lambda row: number(row, "qnn_guided_minus_classical_active_mean"))
    labels = [f"{r['dataset']}/{r['victim']} Q={r['budget']}" for r in rows]
    mean = np.asarray([number(r, "qnn_guided_minus_classical_active_mean") for r in rows])
    low = np.asarray([number(r, "normal_95ci_low") for r in rows])
    high = np.asarray([number(r, "normal_95ci_high") for r in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.4, max(3.5, 0.42 * len(rows) + 1.5)))
    ax.errorbar(mean, y, xerr=np.vstack((mean - low, high - mean)), fmt="o", color="#2f6b5f", ecolor="#555555", capsize=3)
    ax.axvline(0, color="#b54b4b", lw=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Q-Scout-guided clone minus Classical-active clone")
    ax.set_title("Paired multi-seed policy comparison (95% normal CI)")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "fig11_paired_policy_forest.png")
    report.append("fig11 paired forest generated.")


def capacity_noise_heatmaps(rows: list[dict[str, str]], report: list[str]) -> None:
    capacity = [row for row in rows if "capacity" in row.get("source_file", "")]
    noise = [row for row in rows if row.get("noise_kind", "none") != "none"]
    if not capacity and not noise:
        report.append("fig12 heatmaps skipped: no capacity or noise study rows.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, selected, title, xkey, ykey in [
        (axes[0], capacity, "Capacity: QNN fidelity", "qubits", "layers"),
        (axes[1], noise, "Noise: QNN fidelity", "noise_p", "noise_kind"),
    ]:
        if not selected:
            ax.text(0.5, 0.5, "No completed multi-seed study", ha="center", va="center")
            ax.axis("off")
            continue
        grouped = require_multiseed(selected, (xkey, ykey))
        xs, ys = sorted({key[0] for key in grouped}), sorted({key[1] for key in grouped})
        matrix = np.full((len(ys), len(xs)), np.nan)
        for (x, y), group in grouped.items():
            metrics = np.asarray([number(r, "qnn_agreement") for r in group])
            matrix[ys.index(y), xs.index(x)] = np.nanmean(metrics)
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(xs)), xs)
        ax.set_yticks(range(len(ys)), ys)
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
        ax.set_title(title)
        for iy in range(len(ys)):
            for ix in range(len(xs)):
                if np.isfinite(matrix[iy, ix]):
                    ax.text(ix, iy, f"{matrix[iy, ix]:.2f}", ha="center", va="center", color="white" if matrix[iy, ix] < 0.55 else "black")
        fig.colorbar(image, ax=ax, fraction=0.046)
    save(fig, "fig12_capacity_noise_heatmaps.png")
    report.append("fig12 capacity/noise heatmaps generated where multi-seed data exist.")


def defense_cost_3d(rows: list[dict[str, str]], report: list[str]) -> None:
    selected = [row for row in rows if number(row, "defense_label_noise") > 0 and math.isfinite(number(row, "total_seconds"))]
    groups = require_multiseed(selected, ("strategy", "defense_label_noise"))
    if not groups:
        report.append("fig13 defense-cost curve skipped: no multi-seed defense rows with timing.")
        return
    fig = plt.figure(figsize=(8, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"random": "#8d5a35", "classical_active": "#3b6f9c", "active": "#2f6b5f"}
    for (strategy, strength), group in groups.items():
        fidelity = np.nanmean([clone_metric(row) for row in group])
        cost = np.nanmean([number(row, "total_seconds") for row in group])
        ax.scatter(float(strength), fidelity, cost, s=52, color=colors.get(strategy, "#555555"), label=strategy)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False)
    ax.set_xlabel("Label randomization probability")
    ax.set_ylabel("Clone fidelity")
    ax.set_zlabel("Total seconds")
    ax.set_title("Defense strength - fidelity - cost")
    save(fig, "fig13_defense_fidelity_cost_3d.png")
    report.append("fig13 defense/fidelity/cost 3D plot generated.")


def appendix(rows: list[dict[str, str]], report: list[str]) -> None:
    failures = []
    for row in rows:
        reasons = []
        if number(row, "victim_accuracy") < 0.8:
            reasons.append("victim_below_quality_threshold")
        if number(row, "qnn_gain_over_majority") <= 0:
            reasons.append("qnn_not_above_majority")
        if row.get("passes_classical_gate", "True") == "False":
            reasons.append("qnn_below_matched_mlp")
        if reasons:
            failures.append({**row, "failure_reasons": ";".join(reasons)})
    path = OUT / "appendix_failed_units.csv"
    if failures:
        fields = sorted({key for row in failures for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(failures)
    report.append(f"appendix failures: {len(failures)} rows written to {path.name}.")


def main() -> None:
    rows = collect_rows()
    report = [f"multi-seed rows discovered: {len(rows)}"]
    fidelity_query(rows, report)
    paired_forest(report)
    capacity_noise_heatmaps(rows, report)
    defense_cost_3d(rows, report)
    appendix(rows, report)
    (OUT / "main_figure_generation_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
