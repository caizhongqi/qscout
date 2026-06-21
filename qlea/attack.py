"""Hard-label quantum extraction attack against a LoRA-adapted target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qnn import QuantumClassifier, QuantumFeatureMap
from .target import LoRATarget


def rank_project(matrix: np.ndarray, rank: int) -> np.ndarray:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vt[:rank]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1)
    bv = b.reshape(-1)
    return float((av @ bv) / ((np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-12))


@dataclass
class QuantumLoRAExtractor:
    n_qubits: int = 4
    n_layers: int = 2
    rank: int = 2
    noise_kind: str = "none"
    noise_p: float = 0.0
    seed: int = 13

    def fit(
        self,
        target: LoRATarget,
        query_x: np.ndarray,
        *,
        epochs: int = 35,
    ) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        hard_labels = target.predict(query_x)
        fmap = QuantumFeatureMap(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            noise_kind=self.noise_kind,
            noise_p=self.noise_p,
        )
        qnn = QuantumClassifier.create(
            fmap, n_classes=target.base_weight.shape[1], rng=rng
        )
        history = qnn.fit(query_x, hard_labels, epochs=epochs, seed=self.seed + 1)

        correction_logits = qnn.logits(query_x) - query_x @ target.base_weight
        ridge = 1e-2 * np.eye(query_x.shape[1])
        dense_delta = np.linalg.solve(
            query_x.T @ query_x + ridge, query_x.T @ correction_logits
        )
        recovered_delta = rank_project(dense_delta, self.rank)
        return {
            "qnn": qnn,
            "history": history,
            "labels": hard_labels,
            "recovered_delta": recovered_delta,
        }

    def evaluate(
        self,
        target: LoRATarget,
        result: dict[str, object],
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        qnn = result["qnn"]
        recovered_delta = result["recovered_delta"]
        assert isinstance(qnn, QuantumClassifier)
        assert isinstance(recovered_delta, np.ndarray)

        victim_pred = target.predict(x_test)
        qnn_pred = qnn.predict(x_test)
        recovered_pred = np.argmax(
            x_test @ (target.base_weight + recovered_delta), axis=1
        )
        true_delta = target.delta
        return {
            "victim_accuracy": float(np.mean(victim_pred == y_test)),
            "qnn_extraction_accuracy": float(np.mean(qnn_pred == victim_pred)),
            "recovered_model_agreement": float(np.mean(recovered_pred == victim_pred)),
            "delta_cosine": cosine_similarity(true_delta, recovered_delta),
            "delta_relative_error": float(
                np.linalg.norm(true_delta - recovered_delta)
                / (np.linalg.norm(true_delta) + 1e-12)
            ),
        }


@dataclass
class GeneralQuantumExtractor:
    n_qubits: int = 3
    n_layers: int = 2
    noise_kind: str = "none"
    noise_p: float = 0.0
    entanglement: str = "circular"
    data_reuploading: bool = True
    measure_zz: bool = True
    feature_cycling: bool = True
    data_entanglement: bool = True
    seed: int = 13

    def fit(
        self,
        predict_fn,
        query_x: np.ndarray,
        *,
        n_classes: int,
        epochs: int = 12,
    ) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        hard_labels = predict_fn(query_x)
        fmap = QuantumFeatureMap(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            noise_kind=self.noise_kind,
            noise_p=self.noise_p,
            entanglement=self.entanglement,
            data_reuploading=self.data_reuploading,
            measure_zz=self.measure_zz,
            feature_cycling=self.feature_cycling,
            data_entanglement=self.data_entanglement,
        )
        qnn = QuantumClassifier.create(fmap, n_classes=n_classes, rng=rng)
        history = qnn.fit(query_x, hard_labels, epochs=epochs, seed=self.seed + 1)
        return {"qnn": qnn, "history": history, "labels": hard_labels}

    def fit_from_labels(
        self,
        query_x: np.ndarray,
        hard_labels: np.ndarray,
        *,
        n_classes: int,
        epochs: int = 12,
    ) -> dict[str, object]:
        rng = np.random.default_rng(self.seed)
        fmap = QuantumFeatureMap(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            noise_kind=self.noise_kind,
            noise_p=self.noise_p,
            entanglement=self.entanglement,
            data_reuploading=self.data_reuploading,
            measure_zz=self.measure_zz,
            feature_cycling=self.feature_cycling,
            data_entanglement=self.data_entanglement,
        )
        qnn = QuantumClassifier.create(fmap, n_classes=n_classes, rng=rng)
        history = qnn.fit(query_x, hard_labels, epochs=epochs, seed=self.seed + 1)
        return {"qnn": qnn, "history": history, "labels": hard_labels}

    def evaluate(
        self,
        predict_fn,
        result: dict[str, object],
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        qnn = result["qnn"]
        assert isinstance(qnn, QuantumClassifier)
        victim_pred = predict_fn(x_test)
        qnn_pred = qnn.predict(x_test)
        return {
            "victim_accuracy": float(np.mean(victim_pred == y_test)),
            "qnn_extraction_accuracy": float(np.mean(qnn_pred == victim_pred)),
            "recovered_model_agreement": float("nan"),
            "delta_cosine": float("nan"),
            "delta_relative_error": float("nan"),
        }
