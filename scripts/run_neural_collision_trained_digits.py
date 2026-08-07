"""Cross-architecture neural collision experiment on sklearn Digits.

This script gives the paper a fully reproducible trained-network experiment
without an external dataset download.  It trains an MLP and a small CNN on the
8x8 Digits dataset, applies global magnitude pruning, and measures:

* task accuracy;
* local representation Jacobian rank and CDR;
* MLP structural path number nu and numerical-rank equality;
* directly verified collision pairs when nullity is positive.

The MLP path theorem is used only for independently parameterized dense edges.
For the CNN, parameter sharing invalidates that generic-path identification, so
we report the numerical Jacobian certificate only.
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from qlea.neural_collision.core import analyze_relu_mlp, construct_collision
from qlea.neural_collision.structure import structural_path_rank
from qlea.neural_collision.torch_adapter import (
    analyze_torch_module,
    construct_torch_collision,
)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required; install qscout[llm] or qscout[all]") from exc
    return torch


def _set_seed(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _make_models():
    torch = _torch()

    class MLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Linear(64, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Linear(128, 10)

        def forward(self, x):
            x = x.reshape(x.shape[0], -1)
            return self.head(self.features(x))

    class CNNFeatures(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )
            self.proj = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(16 * 8 * 8, 128),
                torch.nn.ReLU(),
            )

        def forward(self, x):
            return self.proj(self.conv(x))

    class CNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = CNNFeatures()
            self.head = torch.nn.Linear(128, 10)

        def forward(self, x):
            return self.head(self.features(x))

    return MLP(), CNN()


def _load_data(seed: int):
    torch = _torch()
    dataset = load_digits()
    x = dataset.images.astype(np.float32) / 16.0
    y = dataset.target.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )
    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )


def _model_input(x, architecture: str):
    if architecture == "MLP":
        return x.reshape(x.shape[0], -1)
    return x[:, None, :, :]


def _train(model, architecture: str, x_train, y_train, *, epochs: int, seed: int, device):
    torch = _torch()
    model.to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = _model_input(xb, architecture).to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _accuracy(model, architecture: str, x_test, y_test, device) -> float:
    torch = _torch()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(x_test), 256):
            xb = _model_input(x_test[start : start + 256], architecture).to(device)
            yb = y_test[start : start + 256].to(device)
            prediction = model(xb).argmax(dim=1)
            correct += int((prediction == yb).sum().item())
            total += int(yb.numel())
    return correct / total


def _global_magnitude_prune(model, sparsity: float):
    torch = _torch()
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must lie in [0, 1)")
    pruned = copy.deepcopy(model)
    parameters = [parameter for parameter in pruned.parameters() if parameter.ndim >= 2]
    if not parameters or sparsity == 0.0:
        return pruned
    values = torch.cat([parameter.detach().abs().reshape(-1).cpu() for parameter in parameters])
    prune_count = int(round(sparsity * values.numel()))
    if prune_count <= 0:
        return pruned
    prune_count = min(prune_count, values.numel() - 1)
    threshold = torch.kthvalue(values, prune_count).values.item()
    with torch.no_grad():
        for parameter in parameters:
            mask = parameter.abs() > threshold
            parameter.mul_(mask)
    return pruned


def _mlp_numpy_layers(model):
    first = model.features[0]
    second = model.features[2]
    return (
        [
            first.weight.detach().cpu().numpy().astype(float),
            second.weight.detach().cpu().numpy().astype(float),
        ],
        [
            first.bias.detach().cpu().numpy().astype(float),
            second.bias.detach().cpu().numpy().astype(float),
        ],
    )


def _analyze_mlp_instances(model, x_test, sparsity: float, limit: int) -> tuple[list[dict], list[dict]]:
    weights, biases = _mlp_numpy_layers(model)
    supports = [np.abs(weight) > 0.0 for weight in weights]
    rows: list[dict] = []
    collisions: list[dict] = []

    for index in range(min(limit, len(x_test))):
        x = x_test[index].reshape(-1).numpy().astype(float)
        certificate = analyze_relu_mlp(
            weights,
            biases,
            x,
            relu_after=[True, True],
            rank_tolerance=1e-9,
        )
        vertex_masks = [
            np.ones(64, dtype=bool),
            certificate.activation_masks[0],
            certificate.activation_masks[1],
        ]
        nu = structural_path_rank(supports, vertex_masks=vertex_masks)
        rows.append(
            {
                "arch": "MLP",
                "sparsity": sparsity,
                "sample": index,
                "rank": certificate.rank,
                "nu": nu,
                "cdr": certificate.collision_deficiency_ratio,
                "rank_equals_nu": int(certificate.rank == nu),
                "active_h1": int(np.count_nonzero(certificate.activation_masks[0])),
                "active_h2": int(np.count_nonzero(certificate.activation_masks[1])),
            }
        )
        if certificate.has_collision_direction:
            witness = construct_collision(
                weights,
                biases,
                x,
                certificate,
                relu_after=[True, True],
                target_l2_shift=0.05,
            )
            collisions.append(
                {
                    "arch": "MLP",
                    "sparsity": sparsity,
                    "sample": index,
                    "same_activation": int(witness.same_activation_region),
                    "l2_shift": witness.l2_shift,
                    "output_error_l2": witness.output_error_l2,
                    "output_error_linf": witness.output_error_linf,
                }
            )
    return rows, collisions


def _analyze_cnn_instances(model, x_test, sparsity: float, limit: int, device) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    collisions: list[dict] = []
    feature_module = model.features
    feature_module.to(device)
    feature_module.eval()

    for index in range(min(limit, len(x_test))):
        x = x_test[index : index + 1, None, :, :].to(device)
        certificate = analyze_torch_module(
            feature_module,
            x,
            rank_tolerance=1e-7,
            vectorize=True,
        )
        rows.append(
            {
                "arch": "CNN",
                "sparsity": sparsity,
                "sample": index,
                "rank": certificate.rank,
                "nu": np.nan,
                "cdr": certificate.collision_deficiency_ratio,
                "rank_equals_nu": np.nan,
                "active_h1": np.nan,
                "active_h2": np.nan,
            }
        )
        if certificate.has_null_direction:
            witness = construct_torch_collision(
                feature_module,
                x,
                certificate,
                target_l2_shift=0.05,
                output_atol=2e-6,
                output_rtol=2e-6,
                max_backtracking_steps=20,
            )
            if witness.numerically_verified:
                collisions.append(
                    {
                        "arch": "CNN",
                        "sparsity": sparsity,
                        "sample": index,
                        "same_activation": witness.same_relu_signature,
                        "l2_shift": witness.l2_shift,
                        "output_error_l2": witness.output_error_l2,
                        "output_error_linf": witness.output_error_linf,
                    }
                )
    return rows, collisions


def _summarize(instance_frame: pd.DataFrame, accuracies: dict[tuple[str, float], float]) -> pd.DataFrame:
    rows = []
    for (arch, sparsity), group in instance_frame.groupby(["arch", "sparsity"]):
        equality = group["rank_equals_nu"].dropna()
        rows.append(
            {
                "arch": arch,
                "sparsity": sparsity,
                "accuracy": accuracies[(arch, float(sparsity))],
                "mean_rank": group["rank"].mean(),
                "min_rank": group["rank"].min(),
                "max_rank": group["rank"].max(),
                "mean_nu": group["nu"].mean(),
                "equality_rate": equality.mean() if len(equality) else np.nan,
                "mean_cdr": group["cdr"].mean(),
                "active_h1": group["active_h1"].mean(),
                "active_h2": group["active_h2"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["arch", "sparsity"])


def _plots(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for arch, group in summary.groupby("arch"):
        ax.plot(group["sparsity"], group["mean_rank"], marker="o", label=arch)
    ax.set_xlabel("global weight sparsity")
    ax.set_ylabel("mean local representation rank")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "crossarch_rank_path.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for arch, group in summary.groupby("arch"):
        ax.plot(group["sparsity"], group["mean_cdr"], marker="o", label=arch)
    ax.set_xlabel("global weight sparsity")
    ax.set_ylabel("mean collision deficiency ratio (CDR)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "crossarch_cdr.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for arch, group in summary.groupby("arch"):
        ax.plot(group["sparsity"], group["accuracy"], marker="o", label=arch)
    ax.set_xlabel("global weight sparsity")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "crossarch_accuracy.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsities", default="0,0.5,0.7,0.8,0.87,0.9,0.93,0.95,0.97,0.98")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--instances", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/neural_collision_trained_digits")
    args = parser.parse_args()

    torch = _torch()
    _set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )
    sparsities = [float(value) for value in args.sparsities.split(",") if value.strip()]
    x_train, y_train, x_test, y_test = _load_data(args.seed)
    mlp, cnn = _make_models()
    mlp = _train(mlp, "MLP", x_train, y_train, epochs=args.epochs, seed=args.seed, device=device)
    cnn = _train(cnn, "CNN", x_train, y_train, epochs=args.epochs, seed=args.seed + 1, device=device)

    instance_rows: list[dict] = []
    collision_rows: list[dict] = []
    accuracies: dict[tuple[str, float], float] = {}

    for architecture, trained in (("MLP", mlp), ("CNN", cnn)):
        for sparsity in sparsities:
            pruned = _global_magnitude_prune(trained, sparsity).to(device)
            accuracies[(architecture, sparsity)] = _accuracy(
                pruned, architecture, x_test, y_test, device
            )
            if architecture == "MLP":
                rows, collisions = _analyze_mlp_instances(
                    pruned, x_test, sparsity, args.instances
                )
            else:
                rows, collisions = _analyze_cnn_instances(
                    pruned, x_test, sparsity, args.instances, device
                )
            instance_rows.extend(rows)
            collision_rows.extend(collisions)

    instance_frame = pd.DataFrame(instance_rows)
    collision_frame = pd.DataFrame(collision_rows)
    summary = _summarize(instance_frame, accuracies)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_frame[instance_frame["arch"] == "MLP"].to_csv(
        output_dir / "mlp_instance_results.csv", index=False
    )
    instance_frame[instance_frame["arch"] == "CNN"].to_csv(
        output_dir / "cnn_instance_results.csv", index=False
    )
    summary[summary["arch"] == "MLP"].to_csv(output_dir / "mlp_summary.csv", index=False)
    summary[summary["arch"] == "CNN"].to_csv(output_dir / "cnn_summary.csv", index=False)
    collision_frame.to_csv(output_dir / "crossarch_exact_collisions.csv", index=False)
    _plots(summary, output_dir)

    print(summary.to_string(index=False))
    if len(collision_frame):
        collision_summary = collision_frame.groupby("arch").agg(
            n=("sample", "count"),
            max_error=("output_error_linf", "max"),
            median_error=("output_error_linf", "median"),
            median_shift=("l2_shift", "median"),
        )
        print("\nCollision witnesses:")
        print(collision_summary.to_string())
    print(f"\nwrote: {output_dir}")


if __name__ == "__main__":
    main()
