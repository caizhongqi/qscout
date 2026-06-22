"""Query selection for label-only black-box extraction.

The selector follows the useful hard-label principle from query-efficient
generation work: spend queries near the current surrogate boundary while
maintaining coverage of the public candidate pool. It never receives victim
scores or gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attack import GeneralQuantumExtractor
from .qnn import softmax


@dataclass
class ActiveQueryConfig:
    budget: int
    initial_queries: int = 64
    batch_size: int = 64
    candidate_size: int = 384
    # Main paper setting: each VQC committee member is warm-trained enough that
    # disagreement reflects a more mature quantum decision boundary rather than
    # random initialization noise.
    warm_epochs: int = 5
    diversity_weight: float = 0.35
    n_qubits: int = 4
    n_layers: int = 3
    entanglement: str = "circular"
    data_reuploading: bool = True
    measure_zz: bool = True
    feature_cycling: bool = True
    committee_size: int = 2
    committee_disagreement_weight: float = 0.20
    seed: int = 7


def _minmax(values: np.ndarray) -> np.ndarray:
    span = float(values.max() - values.min())
    return (values - values.min()) / (span + 1e-12)


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Row-wise categorical entropy with numerical clipping."""
    clipped = np.clip(probs, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _js_mutual_information(committee_probs: np.ndarray) -> np.ndarray:
    """Jensen-Shannon committee mutual information.

    For candidate x, this computes
        H[p_bar(y|x)] - mean_m H[p_m(y|x)],
    separating epistemic member disagreement from common predictive entropy.
    """
    mean_probs = np.mean(committee_probs, axis=0)
    predictive_entropy = _entropy(mean_probs)
    member_entropy = np.mean(np.stack([_entropy(member_probs) for member_probs in committee_probs]), axis=0)
    return predictive_entropy - member_entropy


def select_hard_label_queries(
    x_pool: np.ndarray,
    label_fn,
    *,
    n_classes: int,
    config: ActiveQueryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Select indices and collect labels using random warm-up plus active batches."""
    if config.budget > len(x_pool):
        raise ValueError("query budget exceeds public candidate pool")
    rng = np.random.default_rng(config.seed)
    selected = list(rng.choice(len(x_pool), size=min(config.initial_queries, config.budget), replace=False))
    labels = list(np.asarray(label_fn(np.asarray(selected, dtype=int)), dtype=int))

    while len(selected) < config.budget:
        remaining = np.setdiff1d(np.arange(len(x_pool)), np.asarray(selected), assume_unique=False)
        candidates = rng.choice(remaining, size=min(config.candidate_size, len(remaining)), replace=False)
        committee_probs = []
        for member in range(config.committee_size):
            extractor = GeneralQuantumExtractor(
                n_qubits=config.n_qubits,
                n_layers=config.n_layers,
                entanglement=config.entanglement,
                data_reuploading=config.data_reuploading,
                measure_zz=config.measure_zz,
                feature_cycling=config.feature_cycling,
                seed=config.seed + len(selected) + 10_007 * member,
            )
            qnn = extractor.fit_from_labels(
                x_pool[selected], np.asarray(labels), n_classes=n_classes, epochs=config.warm_epochs
            )["qnn"]
            committee_probs.append(softmax(qnn.logits(x_pool[candidates])))
        committee_probs_array = np.asarray(committee_probs)
        probs = np.mean(committee_probs_array, axis=0)
        top2 = np.partition(probs, -2, axis=1)[:, -2:]
        uncertainty = 1.0 - (top2[:, 1] - top2[:, 0])
        disagreement = _js_mutual_information(committee_probs_array)
        distances = np.linalg.norm(
            x_pool[candidates, None, :] - x_pool[np.asarray(selected)][None, :, :], axis=2
        ).min(axis=1)
        remaining_weight = 1.0 - config.diversity_weight - config.committee_disagreement_weight
        if remaining_weight < 0:
            raise ValueError("diversity and committee weights must sum to at most one")
        score = (
            remaining_weight * _minmax(uncertainty)
            + config.diversity_weight * _minmax(distances)
            + config.committee_disagreement_weight * _minmax(disagreement)
        )
        take = min(config.batch_size, config.budget - len(selected))
        picked = candidates[np.argsort(score)[-take:]]
        selected.extend(int(i) for i in picked)
        labels.extend(np.asarray(label_fn(np.asarray(picked, dtype=int)), dtype=int))
    return np.asarray(selected), np.asarray(labels)
