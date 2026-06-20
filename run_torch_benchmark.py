from __future__ import annotations

from pathlib import Path
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed in the selected Python interpreter. "
        "In VS Code choose interpreter: D:\\ProgramData\\py2\\python.exe"
    ) from exc

from qlea.attack import GeneralQuantumExtractor
from qlea.torch_victims import (
    ImageCNN,
    SequenceCNN,
    SequenceLSTM,
    make_long_time_series,
    make_shape_images,
    train_torch_victim,
)


OUT_DIR = Path("outputs")


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def save_samples(name: str, x: np.ndarray, y: np.ndarray) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(8, 3))
    for ax, sample, label in zip(axes.ravel(), x[:10], y[:10]):
        if sample.ndim == 3 and sample.shape[1] == sample.shape[2]:
            ax.imshow(sample[0], cmap="gray")
            ax.axis("off")
        else:
            ax.plot(sample[0], linewidth=1.3)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(f"y={label}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_torch_samples.png", dpi=160)
    plt.close(fig)


def classical_surrogates(
    q_query: np.ndarray,
    query_y: np.ndarray,
    q_test: np.ndarray,
    victim_test: np.ndarray,
) -> dict[str, float]:
    logistic = LogisticRegression(max_iter=800, random_state=7)
    logistic.fit(q_query, query_y)
    logistic_pred = logistic.predict(q_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        alpha=1e-3,
        learning_rate_init=3e-3,
        max_iter=220,
        random_state=7,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        mlp.fit(q_query, query_y)
    mlp_pred = mlp.predict(q_test)
    return {
        "logistic_agreement": float(accuracy_score(victim_test, logistic_pred)),
        "classical_mlp_agreement": float(accuracy_score(victim_test, mlp_pred)),
    }


def attack_victim(
    victim,
    bundle,
    *,
    query_budget: int = 512,
    q_features: int = 8,
    noise_kind: str = "none",
    noise_p: float = 0.0,
    epochs: int = 14,
) -> dict[str, float]:
    query_x = bundle.x_train[:query_budget]
    query_y = victim.predict(query_x)
    test_x = bundle.x_test[:1500]
    test_y = bundle.y_test[:1500]
    victim_test = victim.predict(test_x)

    projector = make_pipeline(
        StandardScaler(),
        PCA(n_components=q_features, random_state=7),
        StandardScaler(),
    )
    q_query = projector.fit_transform(flatten(query_x))
    q_test = projector.transform(flatten(test_x))

    baselines = classical_surrogates(q_query, query_y, q_test, victim_test)
    extractor = GeneralQuantumExtractor(
        n_qubits=4,
        n_layers=2,
        noise_kind=noise_kind,
        noise_p=noise_p,
        seed=19,
    )
    result = extractor.fit_from_labels(
        q_query,
        query_y,
        n_classes=int(np.max(bundle.y_train)) + 1,
        epochs=epochs,
    )
    qnn = result["qnn"]
    qnn_pred = qnn.predict(q_test)
    return {
        "victim_accuracy": float(np.mean(victim_test == test_y)),
        "qnn_extraction_accuracy": float(np.mean(qnn_pred == victim_test)),
        **baselines,
        "qnn_minus_logistic": float(
            np.mean(qnn_pred == victim_test) - baselines["logistic_agreement"]
        ),
        "qnn_minus_classical_mlp": float(
            np.mean(qnn_pred == victim_test) - baselines["classical_mlp_agreement"]
        ),
        "query_budget": float(query_budget),
        "noise_p": float(noise_p),
        "train_samples": float(bundle.x_train.shape[0]),
        "test_samples": float(bundle.x_test.shape[0]),
    }


def benchmark_grid() -> tuple[list[int], list[tuple[str, float]], int]:
    if os.environ.get("QLEA_FAST") == "1":
        return [256], [("none", 0.0)], 8
    budgets = [
        int(part.strip())
        for part in os.environ.get("QLEA_QUERY_BUDGETS", "128,512,1024").split(",")
        if part.strip()
    ]
    noise = [
        ("none", 0.0),
        ("phase_flip", 0.005),
        ("phase_flip", 0.01),
        ("amplitude_damping", 0.01),
    ]
    return budgets, noise, 14


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    bundles = [make_shape_images(n_samples=6000), make_long_time_series(n_samples=8000)]
    budgets, noise_settings, qnn_epochs = benchmark_grid()
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        save_samples(bundle.name, bundle.x_train, bundle.y_train)
        n_classes = int(np.max(bundle.y_train)) + 1
        if bundle.name == "shape_images_16x16":
            victims = [
                train_torch_victim(
                    "cnn_shape_images_16x16",
                    ImageCNN(n_classes),
                    bundle,
                    epochs=5,
                )
            ]
        else:
            victims = [
                train_torch_victim(
                    "cnn1d_long_time_series",
                    SequenceCNN(n_classes),
                    bundle,
                    epochs=5,
                ),
                train_torch_victim(
                    "lstm_long_time_series",
                    SequenceLSTM(n_classes),
                    bundle,
                    epochs=5,
                ),
            ]
        for victim in victims:
            for query_budget in budgets:
                for noise_kind, noise_p in noise_settings:
                    metrics = attack_victim(
                        victim,
                        bundle,
                        query_budget=query_budget,
                        noise_kind=noise_kind,
                        noise_p=noise_p,
                        epochs=qnn_epochs,
                    )
                    row = {
                        "dataset": bundle.name,
                        "victim": victim.name,
                        "noise_kind": noise_kind,
                        **metrics,
                    }
                    rows.append(row)
                    print(
                        f"{bundle.name:22s} | {victim.name:24s} | "
                        f"q={query_budget:4d} | noise={noise_kind}@{noise_p:.3f} | "
                        f"victim={metrics['victim_accuracy']:.3f} | "
                        f"qnn={metrics['qnn_extraction_accuracy']:.3f} | "
                        f"mlp={metrics['classical_mlp_agreement']:.3f}"
                    )

    header = [
        "dataset",
        "victim",
        "noise_kind",
        "victim_accuracy",
        "qnn_extraction_accuracy",
        "logistic_agreement",
        "classical_mlp_agreement",
        "qnn_minus_logistic",
        "qnn_minus_classical_mlp",
        "query_budget",
        "noise_p",
        "train_samples",
        "test_samples",
    ]
    csv_path = OUT_DIR / "torch_benchmark_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[col]) for col in header) + "\n")
    print(f"results: {csv_path}")


if __name__ == "__main__":
    main()
