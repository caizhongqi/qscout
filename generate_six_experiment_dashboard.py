"""Summarize the six QScout experiment layers from archived local CSV files.

This script is intentionally conservative: an absent multi-seed CSV is shown
as NOT RUN rather than being represented by a placeholder performance value.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"


def rows(name: str) -> list[dict[str, str]]:
    path = OUT / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return float("nan")


def write_summary(public, torch_rows, active, hardware) -> None:
    lines = [
        "# 六层实验结果清单（仅含已保存的真实输出）",
        "",
        "| 实验层 | 当前数据来源 | 状态 | 可报告结果 | 正确解释 |",
        "|---|---|---|---|---|",
    ]
    lines.append("| 1. 主结果矩阵 | public_datasets_smoke_results_20260618.csv | PARTIAL / 单次 smoke | 10 个 dataset-victim 单元，均为 256 查询 | 无多 seed、无强基线、部分 victim 欠训练；不能作主结果 |")
    lines.append("| 2. 强经典基线 | torch_benchmark_results.csv | PARTIAL | 3 个受控单元包含 Logistic 与 MLP clone | QNN 均未超过 MLP clone；尚无 QEDG 官方复现 |")
    lines.append("| 3. 量子消融 | active_hardlabel_benchmark.csv, qscout_runner_smoke.csv | INCOMPLETE | 仅 3q/1L/32Q 与 5q/3L/256Q pilot | 预算、训练轮数不同，不能形成容量结论 |")
    lines.append("| 4. NISQ 噪声 | hardware_simulation_results.csv | SIMULATOR ONLY | 4 个有限 shots 噪声 replay 单元 | 是模拟器 replay，不是 IBM QPU 结果 |")
    lines.append("| 5. 防御边界 | 无 study_defense_*.csv | NOT RUN | 无 | 不得讨论防御效果 |")
    lines.append("| 6. 泛化与成本 | 无 multi-seed study CSV | NOT RUN | 无 | 不得讨论统计泛化或成本优势 |")
    lines.extend(["", "## 现有 Layer 2 对照结果", "", "| 数据 | Victim | QNN Agreement | Logistic Agreement | MLP Agreement | QNN - MLP |", "|---|---|---:|---:|---:|---:|"])
    for r in torch_rows:
        lines.append(f"| {r['dataset']} | {r['victim']} | {f(r, 'qnn_extraction_accuracy'):.3f} | {f(r, 'logistic_agreement'):.3f} | {f(r, 'classical_mlp_agreement'):.3f} | {f(r, 'qnn_minus_classical_mlp'):.3f} |")
    lines.extend(["", "## 现有 Layer 4 有限 shots 模拟结果", "", "| Shots | Noise | p | Analytic Agreement | Replay Agreement |", "|---:|---|---:|---:|---:|"])
    for r in hardware:
        lines.append(f"| {r['shots']} | {r['noise_kind']} | {r['noise_p']} | {f(r, 'simulator_agreement'):.3f} | {f(r, 'hardware_agreement'):.3f} |")
    (OUT / "six_experiment_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def panel_status(ax, title: str, status: str, detail: str, color: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.68, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.44, status, ha="center", va="center", fontsize=15, fontweight="bold", color=color)
    ax.text(0.5, 0.20, detail, ha="center", va="center", fontsize=9, wrap=True)


def main() -> None:
    public = rows("public_datasets_smoke_results_20260618.csv")
    torch_rows = rows("torch_benchmark_results.csv")
    active = rows("active_hardlabel_benchmark.csv") + rows("qscout_runner_smoke.csv")
    hardware = rows("hardware_simulation_results.csv")
    write_summary(public, torch_rows, active, hardware)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    # Layer 1: public-data smoke, shown as diagnostic only.
    ax = axes[0, 0]
    if public:
        names = [f"{r['dataset']}\n{r['victim'].split('_')[0]}" for r in public]
        x = np.arange(len(public))
        ax.scatter(x, [f(r, "victim_accuracy") for r in public], label="Victim accuracy", color="#3b6f9c")
        ax.scatter(x, [f(r, "qnn_extraction_accuracy") for r in public], label="QNN agreement", color="#b54b4b")
        ax.set_xticks(x, names, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title("Layer 1: public-data smoke")
        ax.legend(fontsize=7, frameon=False)
        ax.text(0.01, 0.02, "PARTIAL: not a multi-seed main result", transform=ax.transAxes, fontsize=8, color="#b54b4b")
        ax.grid(axis="y", alpha=0.2)
    else:
        panel_status(ax, "Layer 1: main matrix", "NOT RUN", "No public-dataset output.", "#b54b4b")
    # Layer 2: strong baseline comparison.
    ax = axes[0, 1]
    if torch_rows:
        x = np.arange(len(torch_rows))
        width = 0.25
        ax.bar(x - width, [f(r, "qnn_extraction_accuracy") for r in torch_rows], width, label="QNN", color="#6b5c9b")
        ax.bar(x, [f(r, "logistic_agreement") for r in torch_rows], width, label="Logistic", color="#8d5a35")
        ax.bar(x + width, [f(r, "classical_mlp_agreement") for r in torch_rows], width, label="MLP", color="#2f6b5f")
        ax.set_xticks(x, [r["victim"].replace("_", "\n") for r in torch_rows], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title("Layer 2: matched classical baselines")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis="y", alpha=0.2)
    else:
        panel_status(ax, "Layer 2: baselines", "NOT RUN", "No baseline CSV.", "#b54b4b")
    # Layer 3: only incomparable pilots.
    ax = axes[0, 2]
    if active:
        labels = [f"{r['qubits']}q/{r['layers']}L\nQ={r['budget']}" for r in active]
        ax.bar(np.arange(len(active)), [f(r, "qnn_agreement") for r in active], color="#6b5c9b")
        ax.axhline(np.mean([f(r, "majority_agreement") for r in active]), color="#9d7a1d", linestyle="--", label="mean majority")
        ax.set_xticks(np.arange(len(active)), labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title("Layer 3: capacity pilots")
        ax.legend(fontsize=7, frameon=False)
        ax.text(0.01, 0.02, "INCOMPLETE: budgets/configs differ", transform=ax.transAxes, fontsize=8, color="#b54b4b")
    else:
        panel_status(ax, "Layer 3: ablation", "NOT RUN", "No capacity sweep.", "#b54b4b")
    # Layer 4: finite-shot noise replay.
    ax = axes[1, 0]
    if hardware:
        x = np.arange(len(hardware))
        ax.bar(x, [f(r, "hardware_agreement") for r in hardware], color="#3b6f9c")
        ax.axhline(f(hardware[0], "simulator_agreement"), color="#8d5a35", linestyle="--", label="analytic reference")
        ax.set_xticks(x, [f"{r['noise_kind']}\n{r['noise_p']}" for r in hardware], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title("Layer 4: finite-shot noise replay")
        ax.legend(fontsize=7, frameon=False)
        ax.text(0.01, 0.02, "SIMULATOR ONLY: no QPU job", transform=ax.transAxes, fontsize=8, color="#b54b4b")
    else:
        panel_status(ax, "Layer 4: NISQ", "NOT RUN", "No noise-replay CSV.", "#b54b4b")
    panel_status(axes[1, 1], "Layer 5: defense boundary", "NOT RUN", "Requires study_defense_*.csv\n(label noise, rate limit, OOD detection).", "#b54b4b")
    panel_status(axes[1, 2], "Layer 6: generalization and cost", "NOT RUN", "Requires multi-seed cross-victim CSV\nwith time, queries, shots, and Q@tau.", "#b54b4b")
    fig.suptitle("QScout Six-Layer Experiment Dashboard: Archived Evidence Only", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "six_experiment_dashboard.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(OUT / "six_experiment_results.md")
    print(FIG / "six_experiment_dashboard.png")


if __name__ == "__main__":
    main()
