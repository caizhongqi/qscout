"""Datasets for tabular, image, and time-series extraction benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetBundle:
    name: str
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    raw_train: np.ndarray | None = None
    raw_test: np.ndarray | None = None


def tabular_low_rank(seed: int = 7) -> DatasetBundle:
    x, y = make_classification(
        n_samples=520,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=seed,
    )
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.35, random_state=seed, stratify=y
    )
    return DatasetBundle("tabular_low_rank", x_train, x_test, y_train, y_test)


def digit_images(seed: int = 7) -> DatasetBundle:
    digits = load_digits()
    images = digits.images.astype(float) / 16.0
    x = images.reshape(images.shape[0], -1)
    y = digits.target.astype(int)
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test, raw_train, raw_test = train_test_split(
        x, y, images, test_size=0.35, random_state=seed, stratify=y
    )
    return DatasetBundle("digits_8x8_images", x_train, x_test, y_train, y_test, raw_train, raw_test)


def synthetic_time_series(seed: int = 7) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_samples = 720
    length = 32
    t = np.linspace(0, 1, length)
    series = []
    labels = []
    for idx in range(n_samples):
        cls = idx % 3
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0.0, 0.12, size=length)
        if cls == 0:
            values = np.sin(2 * np.pi * 2 * t + phase) + noise
        elif cls == 1:
            values = np.sign(np.sin(2 * np.pi * 3 * t + phase)) + noise
        else:
            values = 2.0 * (t - 0.5) + 0.5 * np.sin(2 * np.pi * t + phase) + noise
        series.append(values)
        labels.append(cls)
    raw = np.array(series)
    y = np.array(labels)
    x = StandardScaler().fit_transform(raw)
    x_train, x_test, y_train, y_test, raw_train, raw_test = train_test_split(
        x, y, raw, test_size=0.35, random_state=seed, stratify=y
    )
    return DatasetBundle("synthetic_time_series", x_train, x_test, y_train, y_test, raw_train, raw_test)


def all_datasets(seed: int = 7) -> list[DatasetBundle]:
    return [tabular_low_rank(seed), digit_images(seed), synthetic_time_series(seed)]
