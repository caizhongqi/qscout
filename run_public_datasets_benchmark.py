"""Five-public-dataset benchmark for the improved QNN extraction attack.

Datasets:
- MNIST and FashionMNIST from torchvision.
- FordA, Wafer, ElectricDevices from the UCR/UEA time-series archive.

Run in VS Code with the interpreter D:\\ProgramData\\py2\\python.exe, or:

    & "D:\\ProgramData\\py2\\python.exe" run_public_datasets_benchmark.py
"""

from __future__ import annotations

from pathlib import Path
import os
import argparse
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from qlea.attack import GeneralQuantumExtractor
from qlea.public_datasets import PublicDataset, load_selected_public_datasets
from qlea.public_victims import (
    ImageCNN28,
    ImageMLP,
    TimeSeriesCNN,
    TimeSeriesLSTM,
    train_victim,
)


OUT_DIR = Path("outputs")


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def save_samples(data: PublicDataset) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    n = min(12, data.x_train.shape[0])
    fig, axes = plt.subplots(3, 4, figsize=(8, 5))
    for ax, sample, label in zip(axes.ravel(), data.x_train[:n], data.y_train[:n]):
        if data.kind == "image":
            ax.imshow(sample[0], cmap="gray")
            ax.axis("off")
        else:
            ax.plot(sample[0], linewidth=1.1)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(f"y={label}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{data.name}_public_samples.png", dpi=160)
    plt.close(fig)


def choose_victims(data: PublicDataset):
    if data.kind == "image":
        return [
            ("cnn", ImageCNN28(data.n_classes), 3),
            ("mlp", ImageMLP(data.n_classes), 3),
        ]
    length = data.input_shape[-1]
    return [
        ("cnn1d", TimeSeriesCNN(data.n_classes, length), 6),
        ("lstm", TimeSeriesLSTM(data.n_classes), 6),
    ]


def attack(
    data: PublicDataset,
    victim,
    *,
    query_budget: int = 1024,
    q_features: int = 8,
    qnn_epochs: int = 18,
) -> dict[str, float]:
    qn = min(query_budget, data.x_train.shape[0])
    query_x = data.x_train[:qn]
    query_y = victim.predict(query_x)
    eval_n = min(3000, data.x_test.shape[0])
    test_x = data.x_test[:eval_n]
    test_y = data.y_test[:eval_n]
    victim_pred = victim.predict(test_x)

    projector = make_pipeline(
        StandardScaler(),
        PCA(n_components=min(q_features, flatten(query_x).shape[1]), random_state=7),
        StandardScaler(),
    )
    q_query = projector.fit_transform(flatten(query_x))
    q_test = projector.transform(flatten(test_x))

    extractor = GeneralQuantumExtractor(
        n_qubits=4,
        n_layers=3,
        noise_kind="phase_flip",
        noise_p=0.002,
        seed=23,
    )
    result = extractor.fit_from_labels(
        q_query,
        query_y,
        n_classes=data.n_classes,
        epochs=qnn_epochs,
    )
    qnn = result["qnn"]
    qnn_pred = qnn.predict(q_test)
    return {
        "train_samples": float(data.x_train.shape[0]),
        "test_samples": float(data.x_test.shape[0]),
        "query_budget": float(qn),
        "victim_accuracy": float(np.mean(victim_pred == test_y)),
        "qnn_extraction_accuracy": float(np.mean(qnn_pred == victim_pred)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["MNIST", "FashionMNIST", "FordA", "Wafer", "ElectricDevices"],
        help="Subset of datasets to run.",
    )
    parser.add_argument("--query-budget", type=int, default=1024)
    parser.add_argument("--qnn-epochs", type=int, default=18)
    parser.add_argument("--image-epochs", type=int, default=3)
    parser.add_argument("--ts-epochs", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    datasets = load_selected_public_datasets(args.datasets)
    rows: list[dict[str, object]] = []
    for data in datasets:
        print(
            f"\nDataset {data.name}: train={data.x_train.shape[0]}, "
            f"test={data.x_test.shape[0]}, classes={data.n_classes}, shape={data.input_shape}"
        )
        save_samples(data)
        for family, model, epochs in choose_victims(data):
            victim_name = f"{family}_{data.name}"
            default_epochs = args.image_epochs if data.kind == "image" else args.ts_epochs
            victim = train_victim(victim_name, model, data, epochs=min(epochs, default_epochs))
            metrics = attack(
                data,
                victim,
                query_budget=args.query_budget,
                qnn_epochs=args.qnn_epochs,
            )
            row = {"dataset": data.name, "victim": victim_name, **metrics}
            rows.append(row)
            print(
                f"{victim_name:30s} | victim_acc={metrics['victim_accuracy']:.3f} | "
                f"qnn_agreement={metrics['qnn_extraction_accuracy']:.3f}"
            )

    header = [
        "dataset",
        "victim",
        "train_samples",
        "test_samples",
        "query_budget",
        "victim_accuracy",
        "qnn_extraction_accuracy",
    ]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"public_datasets_benchmark_results_{stamp}.csv"
    latest_path = OUT_DIR / "public_datasets_benchmark_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[col]) for col in header) + "\n")
    with latest_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[col]) for col in header) + "\n")
    print(f"\nresults: {csv_path}")


if __name__ == "__main__":
    main()
