"""Generate publication figures from QScout CSV outputs without inventing data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=240, bbox_inches="tight")
    fig.savefig(FIG / name.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)


def experiment_matrix() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")
    columns = ["Datasets", "Victims", "Query policies", "Budgets", "Evidence"]
    entries = [
        ["MNIST", "FashionMNIST", "FordA", "Wafer", "ElectricDevices"],
        ["CNN", "MLP", "1D-CNN", "LSTM"],
        ["Random", "Classical-active", "Q-Scout", "QEDG (external)"],
        ["64", "128", "256", "512", "1024"],
        ["Fidelity", "Majority gain", "Paired CI", "Cost", "Defense curve"],
    ]
    colors = ["#2f6b5f", "#8d5a35", "#6b5c9b", "#9d7a1d", "#3b6f9c"]
    for col, title, items, color in zip(range(5), columns, entries, colors):
        x = 0.25 + 2.12 * col
        ax.text(x + 0.8, 5.5, title, ha="center", va="center", fontsize=12, fontweight="bold")
        for row, item in enumerate(items):
            y = 4.65 - 0.84 * row
            box = plt.Rectangle((x, y), 1.62, 0.52, facecolor="white", edgecolor=color, linewidth=1.7)
            ax.add_patch(box)
            ax.text(x + 0.81, y + 0.26, item, ha="center", va="center", fontsize=9)
        if col < 4:
            ax.annotate("", xy=(x + 2.02, 3.1), xytext=(x + 1.7, 3.1), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#333333"})
    ax.text(5.5, 0.35, "All method comparisons share the candidate pool, final clone, budget, and random seed.", ha="center", fontsize=10)
    save(fig, "fig6_experiment_matrix.png")


def pilot_boundary() -> None:
    path = OUT / "active_hardlabel_benchmark.csv"
    if not path.exists():
        return
    rows = read_rows(path)
    if not rows:
        return
    row = rows[0]
    labels = ["Majority", "QNN surrogate", "Random + MLP"]
    values = [float(row["majority_agreement"]), float(row["qnn_agreement"]), float(row["mlp_agreement"])]
    colors = ["#9d7a1d", "#b54b4b", "#2f6b5f"]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Agreement with clean victim")
    ax.set_title("Current MNIST pilot: capability boundary, not a main result")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig7_pilot_capability_boundary.png")


def noise_replay() -> None:
    path = OUT / "hardware_simulation_results.csv"
    if not path.exists():
        return
    rows = read_rows(path)
    names = [f"{r['noise_kind']}\n{r['noise_p']}" for r in rows]
    simulator = [float(r["simulator_agreement"]) for r in rows]
    replay = [float(r["hardware_agreement"]) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(x - 0.18, simulator, width=0.36, label="analytic simulator", color="#3b6f9c")
    ax.bar(x + 0.18, replay, width=0.36, label="finite-shot noisy replay", color="#8d5a35")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Agreement")
    ax.set_title("Finite-shot noisy replay (simulator only)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig8_noise_replay_results.png")


def reporting_gates() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    steps = [
        ("Victim quality", "accuracy >= threshold"),
        ("Extraction validity", "above majority baseline"),
        ("Fair comparison", "matched clone / budget / seed"),
        ("Statistical evidence", "paired CI excludes zero"),
        ("Hardware evidence", "job ID and calibration"),
    ]
    for idx, (title, detail) in enumerate(steps):
        x = 0.15 + idx * 1.98
        ax.add_patch(plt.Rectangle((x, 1.15), 1.58, 1.45, facecolor="#f7f7f7", edgecolor="#444444", linewidth=1.3))
        ax.text(x + 0.79, 2.14, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 0.79, 1.62, detail, ha="center", va="center", fontsize=8.5, wrap=True)
        if idx < len(steps) - 1:
            ax.annotate("", xy=(x + 1.91, 1.88), xytext=(x + 1.62, 1.88), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.text(5, 0.45, "A quantum advantage claim is permitted only after all gates pass.", ha="center", fontsize=10)
    save(fig, "fig9_reporting_gates.png")


if __name__ == "__main__":
    experiment_matrix()
    pilot_boundary()
    noise_replay()
    reporting_gates()
    print(f"figures written to {FIG}")
