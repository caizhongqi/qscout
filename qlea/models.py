"""Classical victim models used in the extraction benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

from .target import LoRATarget


class PredictOnlyModel:
    name: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class LoRAVictim(PredictOnlyModel):
    target: LoRATarget
    name: str = "low_rank_lora_head"

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.target.predict(x)


@dataclass
class SklearnVictim(PredictOnlyModel):
    estimator: object
    name: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.estimator.predict(x)


def fit_lora_victim(x: np.ndarray, y: np.ndarray, rank: int = 2, seed: int = 7) -> LoRAVictim:
    rng = np.random.default_rng(seed)
    n_classes = int(np.max(y)) + 1
    y_onehot = np.eye(n_classes)[y]
    ridge = 1e-2 * np.eye(x.shape[1])
    base_weight = np.linalg.solve(x.T @ x + ridge, x.T @ y_onehot)
    lora_a = rng.normal(0.0, 0.65, size=(rank, x.shape[1]))
    lora_b = rng.normal(0.0, 0.65, size=(n_classes, rank))
    return LoRAVictim(LoRATarget(base_weight=base_weight, lora_a=lora_a, lora_b=lora_b))


def fit_mlp_victim(
    x: np.ndarray,
    y: np.ndarray,
    *,
    name: str,
    hidden: tuple[int, ...] = (64, 32),
    seed: int = 7,
) -> SklearnVictim:
    estimator = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=3e-3,
            max_iter=260,
            random_state=seed,
        ),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        estimator.fit(x, y)
    return SklearnVictim(estimator=estimator, name=name)
