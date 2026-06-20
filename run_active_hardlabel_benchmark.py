"""Multi-seed hard-label extraction benchmark with active query selection.

Example smoke pilot:
  & "D:\\ProgramData\\py2\\python.exe" run_active_hardlabel_benchmark.py `
      --datasets Wafer --victims cnn1d --budgets 128,256 --seeds 7,19
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qlea.attack import GeneralQuantumExtractor
from qlea.public_datasets import PublicDataset, load_selected_public_datasets
from qlea.public_victims import ImageCNN28, ImageMLP, TimeSeriesCNN, TimeSeriesLSTM, train_victim
from qlea.query_strategy import ActiveQueryConfig, select_hard_label_queries


OUT = Path("outputs")


def flat(x: np.ndarray) -> np.ndarray:
    return x.reshape(len(x), -1)


def parse_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_config_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def victims(data: PublicDataset, names: list[str]):
    if data.kind == "image":
        candidates = {
            "cnn": ("cnn", ImageCNN28(data.n_classes)),
            "mlp": ("mlp", ImageMLP(data.n_classes)),
        }
    else:
        length = data.input_shape[-1]
        candidates = {
            "cnn1d": ("cnn1d", TimeSeriesCNN(data.n_classes, length)),
            "lstm": ("lstm", TimeSeriesLSTM(data.n_classes)),
        }
    return [candidates[name] for name in names if name in candidates]


def fit_classical(xq, yq, xt, victim_pred, seed: int) -> tuple[float, float]:
    if np.unique(yq).size < 2:
        constant = int(yq[0])
        agreement = float(np.mean(victim_pred == constant))
        return agreement, agreement
    logistic = LogisticRegression(max_iter=1000, random_state=seed)
    logistic.fit(xq, yq)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        mlp = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300, random_state=seed)
        mlp.fit(xq, yq)
    return float(np.mean(logistic.predict(xt) == victim_pred)), float(np.mean(mlp.predict(xt) == victim_pred))


def select_classical_active_queries(x_pool, label_fn, *, budget: int, n_classes: int, seed: int, initial: int, batch: int, candidates: int) -> tuple[np.ndarray, np.ndarray]:
    """Strong non-quantum active-query control using MLP uncertainty plus diversity."""
    rng = np.random.default_rng(seed)
    selected = list(rng.choice(len(x_pool), size=min(initial, budget), replace=False))
    labels = list(np.asarray(label_fn(np.asarray(selected)), dtype=int))
    while len(selected) < budget:
        remaining = np.setdiff1d(np.arange(len(x_pool)), np.asarray(selected), assume_unique=False)
        candidate = rng.choice(remaining, size=min(candidates, len(remaining)), replace=False)
        if np.unique(labels).size < 2:
            uncertainty = rng.random(len(candidate))
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                guide = MLPClassifier(hidden_layer_sizes=(48,), alpha=1e-3, max_iter=120, random_state=seed + len(selected))
                guide.fit(x_pool[np.asarray(selected)], np.asarray(labels))
            probs = guide.predict_proba(x_pool[candidate])
            top2 = np.partition(probs, -2, axis=1)[:, -2:]
            uncertainty = 1.0 - (top2[:, 1] - top2[:, 0])
        distances = np.linalg.norm(x_pool[candidate, None, :] - x_pool[np.asarray(selected)][None, :, :], axis=2).min(axis=1)
        def norm(v): return (v - v.min()) / (v.max() - v.min() + 1e-12)
        score = 0.65 * norm(uncertainty) + 0.35 * norm(distances)
        take = min(batch, budget - len(selected))
        picked = candidate[np.argsort(score)[-take:]]
        selected.extend(int(i) for i in picked)
        labels.extend(np.asarray(label_fn(picked), dtype=int))
    return np.asarray(selected), np.asarray(labels)


def run_one(data, victim, strategy: str, budget: int, seed: int, args) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    pool_raw = data.x_train
    test_raw = data.x_test[: min(args.eval_samples, len(data.x_test))]
    # The QNN guide sees a compact NISQ-compatible representation, while the
    # final classical clone receives the full public representation. This tests
    # query efficiency rather than artificially bottlenecking every baseline.
    guide_projector = make_pipeline(StandardScaler(), PCA(n_components=args.features, random_state=seed), StandardScaler())
    pool = guide_projector.fit_transform(flat(pool_raw))
    test = guide_projector.transform(flat(test_raw))
    clone_projector = StandardScaler()
    clone_pool = clone_projector.fit_transform(flat(pool_raw))
    clone_test = clone_projector.transform(flat(test_raw))
    victim_pred = victim.predict(test_raw)
    if strategy == "active":
        indices, labels = select_hard_label_queries(
            pool,
            lambda indices: victim.predict(pool_raw[indices]),
            n_classes=data.n_classes,
            config=ActiveQueryConfig(
                budget=budget, initial_queries=min(args.initial_queries, budget), batch_size=args.batch_size,
                candidate_size=args.candidates, warm_epochs=args.warm_epochs, diversity_weight=args.diversity_weight,
                n_qubits=args.qubits, n_layers=args.layers, entanglement=args.entanglement, seed=seed,
            ),
        )
    elif strategy == "classical_active":
        indices, labels = select_classical_active_queries(
            clone_pool, lambda idx: victim.predict(pool_raw[idx]), budget=budget,
            n_classes=data.n_classes, seed=seed, initial=min(args.initial_queries, budget),
            batch=args.batch_size, candidates=args.candidates,
        )
    else:
        indices = rng.choice(len(pool), size=budget, replace=False)
        labels = victim.predict(pool_raw[indices])
    extractor = GeneralQuantumExtractor(
        n_qubits=args.qubits, n_layers=args.layers, entanglement=args.entanglement,
        data_reuploading=True, measure_zz=True, feature_cycling=True,
        noise_kind=args.noise_kind, noise_p=args.noise_p, seed=seed,
    )
    qnn = extractor.fit_from_labels(pool[indices], labels, n_classes=data.n_classes, epochs=args.final_epochs)["qnn"]
    qnn_agreement = float(np.mean(qnn.predict(test) == victim_pred))
    logistic, mlp = fit_classical(clone_pool[indices], labels, clone_test, victim_pred, seed)
    counts = np.bincount(victim_pred, minlength=data.n_classes)
    majority = float(counts.max() / len(victim_pred))
    gain_over_majority = qnn_agreement - majority
    return {
        "dataset": data.name, "victim": victim.name, "strategy": strategy, "seed": seed, "budget": budget,
        "victim_accuracy": float(np.mean(victim_pred == data.y_test[:len(test_raw)])),
        "qnn_agreement": qnn_agreement, "logistic_agreement": logistic, "mlp_agreement": mlp,
        "hybrid_clone_agreement": mlp if strategy == "active" else float("nan"),
        "classical_active_clone_agreement": mlp if strategy == "classical_active" else float("nan"),
        "majority_agreement": majority, "qnn_gain_over_majority": gain_over_majority,
        "qnn_minus_mlp": qnn_agreement - mlp,
        "passes_majority_gate": bool(gain_over_majority >= 0.05),
        "passes_classical_gate": bool(qnn_agreement >= mlp),
        "qubits": args.qubits, "layers": args.layers, "entanglement": args.entanglement,
        "noise_kind": args.noise_kind, "noise_p": args.noise_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Wafer"])
    parser.add_argument("--victims", nargs="+", default=["cnn1d"])
    parser.add_argument("--budgets", default="64,128,256,512,1024")
    parser.add_argument("--seeds", default="7,19,31,43,59")
    parser.add_argument("--strategies", nargs="+", default=["random", "active"])
    parser.add_argument("--victim-epochs", type=int, default=10)
    parser.add_argument("--min-victim-accuracy", type=float, default=0.80)
    parser.add_argument("--final-epochs", type=int, default=14)
    parser.add_argument("--warm-epochs", type=int, default=3)
    parser.add_argument("--initial-queries", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=384)
    parser.add_argument("--diversity-weight", type=float, default=0.35)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--qubits-list", default=None, help="Comma-separated capacity sweep, e.g. 2,3,4,5")
    parser.add_argument("--layers-list", default=None, help="Comma-separated depth sweep, e.g. 1,2,3")
    parser.add_argument("--features-list", default=None, help="Comma-separated encoding sweep, e.g. 8,12,16")
    parser.add_argument("--entanglement", default="circular", choices=["none", "linear", "circular", "star", "full"])
    parser.add_argument("--noise-kind", default="none", choices=["none", "phase_flip", "bit_flip", "amplitude_damping"])
    parser.add_argument("--noise-p", type=float, default=0.0)
    parser.add_argument("--eval-samples", type=int, default=1500)
    parser.add_argument("--output", default="active_hardlabel_benchmark.csv")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    rows = []
    qubit_values = parse_config_list(args.qubits_list) if args.qubits_list else [args.qubits]
    layer_values = parse_config_list(args.layers_list) if args.layers_list else [args.layers]
    feature_values = parse_config_list(args.features_list) if args.features_list else [args.features]
    for data in load_selected_public_datasets(args.datasets):
        for family, model in victims(data, args.victims):
            victim = train_victim(f"{family}_{data.name}_active", model, data, epochs=args.victim_epochs)
            victim_quality = float(np.mean(victim.predict(data.x_test[:min(args.eval_samples, len(data.x_test))]) == data.y_test[:min(args.eval_samples, len(data.x_test))]))
            if victim_quality < args.min_victim_accuracy:
                print(f"SKIP {victim.name}: victim_accuracy={victim_quality:.3f} < {args.min_victim_accuracy:.3f}")
                continue
            for qubits in qubit_values:
                for layers in layer_values:
                    for features in feature_values:
                        args.qubits, args.layers, args.features = qubits, layers, features
                        for budget in parse_list(args.budgets):
                            for seed in parse_list(args.seeds):
                                for strategy in args.strategies:
                                    row = run_one(data, victim, strategy, budget, seed, args)
                                    rows.append(row)
                                    print(row)
    if not rows:
        raise RuntimeError("No valid rows: inspect victim quality thresholds and requested datasets.")
    path = OUT / args.output
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("dataset", "victim", "strategy", "budget", "qubits", "layers", "entanglement", "noise_kind", "noise_p"))
        groups[key].append(row)
    summary_path = path.with_name(path.stem + "_summary.csv")
    fields = ["dataset", "victim", "strategy", "budget", "qubits", "layers", "entanglement", "noise_kind", "noise_p", "runs", "victim_accuracy_mean", "qnn_agreement_mean", "qnn_agreement_std", "mlp_agreement_mean", "mlp_agreement_std", "majority_agreement_mean", "qnn_gain_over_majority_mean"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, values in groups.items():
            qnn = np.array([float(v["qnn_agreement"]) for v in values])
            mlp = np.array([float(v["mlp_agreement"]) for v in values])
            majority = np.array([float(v["majority_agreement"]) for v in values])
            writer.writerow({
                **dict(zip(fields[:9], key)), "runs": len(values),
                "victim_accuracy_mean": float(np.mean([float(v["victim_accuracy"]) for v in values])),
                "qnn_agreement_mean": float(qnn.mean()), "qnn_agreement_std": float(qnn.std(ddof=1)) if len(qnn) > 1 else 0.0,
                "mlp_agreement_mean": float(mlp.mean()), "mlp_agreement_std": float(mlp.std(ddof=1)) if len(mlp) > 1 else 0.0,
                "majority_agreement_mean": float(majority.mean()), "qnn_gain_over_majority_mean": float(np.mean([float(v["qnn_gain_over_majority"]) for v in values]),)
            })
    print(f"results: {path}")
    print(f"summary: {summary_path}")

    # Compare policy-selected classical clones with matched seed/budget cells.
    paired: dict[tuple[object, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["strategy"] not in {"active", "classical_active"}:
            continue
        key = tuple(row[k] for k in ("dataset", "victim", "seed", "budget", "qubits", "layers", "entanglement", "noise_kind", "noise_p"))
        metric = "hybrid_clone_agreement" if row["strategy"] == "active" else "classical_active_clone_agreement"
        paired[key][str(row["strategy"])] = float(row[metric])
    grouped_deltas: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for key, values in paired.items():
        if {"active", "classical_active"}.issubset(values):
            grouped_deltas[key[:2] + key[3:]].append(values["active"] - values["classical_active"])
    comparison_rows = []
    for key, deltas in grouped_deltas.items():
        values = np.asarray(deltas, dtype=float)
        se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        comparison_rows.append({
            **dict(zip(("dataset", "victim", "budget", "qubits", "layers", "entanglement", "noise_kind", "noise_p"), key)),
            "paired_runs": len(values),
            "qnn_guided_minus_classical_active_mean": float(values.mean()),
            "qnn_guided_minus_classical_active_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "normal_95ci_low": float(values.mean() - 1.96 * se),
            "normal_95ci_high": float(values.mean() + 1.96 * se),
            "wins": int(np.sum(values > 0.0)), "ties": int(np.sum(values == 0.0)), "losses": int(np.sum(values < 0.0)),
        })
    comparison_path = path.with_name(path.stem + "_paired_policy_comparison.csv")
    if comparison_rows:
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
            writer.writeheader()
            writer.writerows(comparison_rows)
        print(f"paired policy comparison: {comparison_path}")


if __name__ == "__main__":
    main()
