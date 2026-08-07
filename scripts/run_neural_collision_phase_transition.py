"""Reproduce the sparse-network structural collision phase transition.

Example
-------
python scripts/run_neural_collision_phase_transition.py \
    --dimensions 32,64,128 \
    --c-values -4,-3,-2,-1,0,1,2,3,4 \
    --repetitions 200 \
    --output-dir outputs/neural_collision_phase
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qlea.neural_collision.random_graph import phase_transition_sweep


def _ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def _floats(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def _to_frame(points) -> pd.DataFrame:
    rows = []
    for point in points:
        raw = asdict(point)
        rows.append(
            {
                "d": raw["dimension"],
                "L": raw["layers"],
                "c": raw["c"],
                "p": raw["edge_probability"],
                "reps": raw["repetitions"],
                "pinj": raw["injective_probability"],
                "noninj": raw["noninjective_probability"],
                "residual_rate": raw["residual_failure_rate"],
                "iso_share": raw["isolated_failure_share"],
                "theory": raw["theory_probability"],
                "mean_rank": raw["mean_path_rank"],
                "mean_cdr": raw["mean_cdr"],
            }
        )
    return pd.DataFrame(rows)


def _plot_phase_collapse(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for dimension, group in frame.groupby("d"):
        group = group.sort_values("c")
        ax.plot(group["c"], group["pinj"], marker="o", label=f"d={dimension}")
    theory = frame.groupby("c", as_index=False)["theory"].mean().sort_values("c")
    ax.plot(theory["c"], theory["theory"], color="black", linestyle="--", linewidth=2.0, label="asymptotic theory")
    ax.set_xlabel(r"critical offset $c$ in $p=(\log d+c)/d$")
    ax.set_ylabel("structural injectivity probability")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_residual(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for dimension, group in frame.groupby("d"):
        group = group.sort_values("c")
        axes[0].plot(group["c"], group["residual_rate"], marker="o", label=f"d={dimension}")
        axes[1].plot(group["c"], group["iso_share"], marker="o", label=f"d={dimension}")
    axes[0].set_xlabel("critical offset c")
    axes[0].set_ylabel("failure without isolation")
    axes[1].set_xlabel("critical offset c")
    axes[1].set_ylabel("isolation share among failures")
    axes[1].set_ylim(-0.03, 1.03)
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_cdr(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for dimension, group in frame.groupby("d"):
        group = group.sort_values("c")
        ax.plot(group["c"], group["mean_cdr"], marker="o", label=f"d={dimension}")
    ax.set_xlabel("critical offset c")
    ax.set_ylabel("mean collision deficiency ratio (CDR)")
    ax.set_ylim(bottom=-0.01)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="32,64,128")
    parser.add_argument("--c-values", default="-4,-3,-2,-1,0,1,2,3,4")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output-dir", default="outputs/neural_collision_phase")
    parser.add_argument(
        "--skip-cdr",
        action="store_true",
        help="skip exact layered path-rank averages for a faster injectivity-only sweep",
    )
    args = parser.parse_args()

    dimensions = _ints(args.dimensions)
    c_values = _floats(args.c_values)
    if not dimensions or not c_values:
        raise SystemExit("dimensions and c-values must be non-empty")

    points = phase_transition_sweep(
        dimensions,
        c_values,
        layers=args.layers,
        repetitions=args.repetitions,
        seed=args.seed,
        compute_mean_path_rank=not args.skip_cdr,
    )
    frame = _to_frame(points)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "phase_transition_sweep.csv", index=False)
    _plot_phase_collapse(frame, output_dir / "phase_transition_collapse.png")
    _plot_residual(frame, output_dir / "critical_residual_rate.png")
    if not args.skip_cdr:
        _plot_cdr(frame, output_dir / "cdr_transition.png")

    summary = frame.groupby("d", as_index=False).agg(
        min_pinj=("pinj", "min"),
        max_pinj=("pinj", "max"),
        max_residual=("residual_rate", "max"),
        mean_iso_share=("iso_share", "mean"),
    )
    print(summary.to_string(index=False))
    print(f"wrote: {output_dir}")


if __name__ == "__main__":
    main()
