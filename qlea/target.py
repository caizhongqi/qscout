"""Classical low-rank adapted target model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class LoRATarget:
    base_weight: np.ndarray
    lora_a: np.ndarray
    lora_b: np.ndarray
    alpha: float = 1.0

    @property
    def delta(self) -> np.ndarray:
        rank = self.lora_a.shape[0]
        return (self.alpha / rank) * (self.lora_a.T @ self.lora_b.T)

    @property
    def weight(self) -> np.ndarray:
        return self.base_weight + self.delta

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(x), axis=1)


def make_synthetic_task(
    *,
    n_samples: int = 900,
    n_features: int = 8,
    n_classes: int = 3,
    rank: int = 2,
    seed: int = 7,
) -> tuple[LoRATarget, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(6, n_features),
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=seed,
    )
    x = StandardScaler().fit_transform(x)
    y_onehot = np.eye(n_classes)[y]
    ridge = 1e-2 * np.eye(n_features)
    base_weight = np.linalg.solve(x.T @ x + ridge, x.T @ y_onehot)

    lora_a = rng.normal(0.0, 0.65, size=(rank, n_features))
    lora_b = rng.normal(0.0, 0.65, size=(n_classes, rank))
    target = LoRATarget(base_weight=base_weight, lora_a=lora_a, lora_b=lora_b)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.35, random_state=seed, stratify=y
    )
    return target, x_train, x_test, y_train, y_test
