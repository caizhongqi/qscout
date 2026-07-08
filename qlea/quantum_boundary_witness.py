"""Quantum state-discrimination boundary witness utilities.

The scorer is a lightweight density-matrix layer for query selection.  It does
not assume hardware speedup; it gives QScout a concrete quantum information
object: empirical state densities, fidelity margins, and a Helstrom-inspired
positive-vs-negative witness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantumBoundaryWitnessResult:
    fidelity_top1: np.ndarray
    fidelity_top2: np.ndarray
    fidelity_margin: np.ndarray
    boundary_score: np.ndarray
    positive_evidence: np.ndarray
    helstrom_score: np.ndarray
    helstrom_boundary: np.ndarray
    reliability: np.ndarray
    final_acquisition: np.ndarray


class QuantumBoundaryWitness:
    """Density-matrix and Helstrom-style scorer for candidate states."""

    def __init__(self, *, regularization: float = 1e-3) -> None:
        self.regularization = float(regularization)
        self.class_densities_: list[np.ndarray] = []
        self.positive_density_: np.ndarray | None = None
        self.negative_density_: np.ndarray | None = None
        self.prior_positive_: float = 0.5
        self.empirical_reliability_: float = 0.0
        self.dim_: int = 0

    def fit(
        self,
        states: np.ndarray,
        *,
        observed_indices: list[int] | np.ndarray,
        utilities: list[float] | np.ndarray,
        prior: np.ndarray | None = None,
        class_labels: list[int] | np.ndarray | None = None,
    ) -> "QuantumBoundaryWitness":
        states = _normalize_states(states)
        self.dim_ = int(states.shape[1]) if states.ndim == 2 else 0
        if self.dim_ == 0:
            self.class_densities_ = []
            self.positive_density_ = None
            self.negative_density_ = None
            self.empirical_reliability_ = 0.0
            return self

        observed = np.asarray(observed_indices, dtype=int)
        utilities_array = np.asarray(utilities, dtype=float)
        valid = (observed >= 0) & (observed < len(states))
        observed = observed[valid]
        utilities_array = utilities_array[valid]

        if class_labels is None:
            labels = _utility_classes(utilities_array)
        else:
            labels = np.asarray(class_labels, dtype=int)[valid]

        densities: list[np.ndarray] = []
        for label in sorted(set(labels.tolist())):
            indices = observed[labels == label]
            if len(indices) > 0:
                densities.append(_density_matrix(states, indices, self.regularization))

        empirical_classes = len(densities)
        if empirical_classes < 2:
            fallback = _fallback_density_pairs(states, prior)
            densities = densities + fallback[: max(0, 2 - empirical_classes)]

        self.class_densities_ = densities
        self.empirical_reliability_ = min(
            1.0,
            0.30 + 0.12 * float(len(observed)) + 0.18 * float(max(empirical_classes - 1, 0)),
        )

        pos_indices, neg_indices = _positive_negative_indices(observed, utilities_array, states, prior)
        self.positive_density_ = _density_matrix(states, pos_indices, self.regularization)
        self.negative_density_ = _density_matrix(states, neg_indices, self.regularization)
        self.prior_positive_ = float(len(pos_indices) / max(len(pos_indices) + len(neg_indices), 1))
        self.prior_positive_ = float(np.clip(self.prior_positive_, 0.15, 0.85))
        return self

    def score(self, states: np.ndarray, reliability_signal: np.ndarray | None = None) -> QuantumBoundaryWitnessResult:
        states = _normalize_states(states)
        n = len(states)
        if n == 0:
            z = np.zeros(0, dtype=float)
            return QuantumBoundaryWitnessResult(z, z, z, z, z, z, z, z, z)

        if not self.class_densities_:
            self.class_densities_ = _fallback_density_pairs(states, None)
            self.empirical_reliability_ = 0.25
        fidelities = np.column_stack([_pure_to_density_fidelity(states, rho) for rho in self.class_densities_])
        if fidelities.shape[1] == 1:
            fidelities = np.column_stack([fidelities[:, 0], 1.0 - fidelities[:, 0]])
        sorted_f = np.sort(fidelities, axis=1)
        top1 = sorted_f[:, -1]
        top2 = sorted_f[:, -2]
        margin = np.clip(top1 - top2, 0.0, 1.0)
        boundary = _minmax(top2 * (1.0 - _minmax(margin)))

        if self.positive_density_ is None or self.negative_density_ is None:
            self.positive_density_, self.negative_density_ = _fallback_density_pairs(states, None)
        positive = _pure_to_density_fidelity(states, self.positive_density_)
        negative = _pure_to_density_fidelity(states, self.negative_density_)
        delta = self.prior_positive_ * self.positive_density_ - (1.0 - self.prior_positive_) * self.negative_density_
        helstrom_raw = np.real(np.einsum("bi,ij,bj->b", states.conj(), delta, states))
        helstrom_norm = _signed_minmax(helstrom_raw)
        helstrom_boundary = 1.0 - _minmax(np.abs(helstrom_norm))

        if reliability_signal is None:
            reliability = np.ones(n, dtype=float)
        else:
            rel = np.asarray(reliability_signal, dtype=float)
            reliability = 1.0 / (1.0 + _minmax(rel))
        reliability = np.clip(reliability * max(self.empirical_reliability_, 0.20), 0.0, 1.0)

        final = (
            0.34 * _minmax(positive)
            + 0.27 * boundary
            + 0.23 * _minmax(helstrom_norm)
            + 0.16 * helstrom_boundary
        ) * reliability

        return QuantumBoundaryWitnessResult(
            fidelity_top1=np.asarray(top1, dtype=float),
            fidelity_top2=np.asarray(top2, dtype=float),
            fidelity_margin=np.asarray(margin, dtype=float),
            boundary_score=np.asarray(boundary, dtype=float),
            positive_evidence=np.asarray(positive, dtype=float),
            helstrom_score=np.asarray(helstrom_norm, dtype=float),
            helstrom_boundary=np.asarray(helstrom_boundary, dtype=float),
            reliability=np.asarray(reliability, dtype=float),
            final_acquisition=_minmax(final),
        )


def _normalize_states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.complex128)
    if states.ndim != 2:
        return np.zeros((0, 0), dtype=np.complex128)
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    return states / (norms + 1e-12)


def _density_matrix(states: np.ndarray, indices: np.ndarray, regularization: float) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    indices = indices[(indices >= 0) & (indices < len(states))]
    dim = int(states.shape[1])
    if len(indices) == 0:
        return np.eye(dim, dtype=np.complex128) / float(dim)
    selected = states[indices]
    rho = selected.conj().T @ selected / float(len(selected))
    if regularization > 0:
        rho = (1.0 - regularization) * rho + regularization * np.eye(dim, dtype=np.complex128) / float(dim)
    trace = float(np.real(np.trace(rho)))
    if trace <= 1e-12:
        return np.eye(dim, dtype=np.complex128) / float(dim)
    return rho / trace


def _pure_to_density_fidelity(states: np.ndarray, rho: np.ndarray) -> np.ndarray:
    values = np.einsum("bi,ij,bj->b", states.conj(), rho, states)
    return np.clip(np.real(values), 0.0, 1.0)


def _fallback_density_pairs(states: np.ndarray, prior: np.ndarray | None) -> list[np.ndarray]:
    n = len(states)
    if n == 0:
        return []
    if prior is None or len(prior) != n:
        scores = np.arange(n, dtype=float)
    else:
        scores = np.asarray(prior, dtype=float)
    order = np.argsort(scores)
    take = max(1, min(4, n // 5 or 1))
    return [
        _density_matrix(states, order[:take], 0.0),
        _density_matrix(states, order[-take:], 0.0),
    ]


def _utility_classes(utilities: np.ndarray) -> np.ndarray:
    if len(utilities) == 0:
        return np.zeros(0, dtype=int)
    if float(np.max(utilities) - np.min(utilities)) <= 1e-12:
        return np.zeros(len(utilities), dtype=int)
    q1, q2 = np.quantile(utilities, [0.34, 0.67])
    return np.asarray((utilities > q1).astype(int) + (utilities > q2).astype(int), dtype=int)


def _positive_negative_indices(
    observed: np.ndarray,
    utilities: np.ndarray,
    states: np.ndarray,
    prior: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(observed) >= 2 and float(np.max(utilities) - np.min(utilities)) > 1e-12:
        threshold = float(np.median(utilities))
        pos = observed[utilities >= threshold]
        neg = observed[utilities < threshold]
        if len(pos) > 0 and len(neg) > 0:
            return pos, neg
    fallback = _fallback_indices(len(states), prior)
    return fallback


def _fallback_indices(n: int, prior: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if n == 0:
        z = np.zeros(0, dtype=int)
        return z, z
    if prior is None or len(prior) != n:
        scores = np.arange(n, dtype=float)
    else:
        scores = np.asarray(prior, dtype=float)
    order = np.argsort(scores)
    take = max(1, min(4, n // 5 or 1))
    return order[-take:], order[:take]


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    span = float(np.nanmax(values) - np.nanmin(values))
    if not np.isfinite(span) or span <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.nanmin(values))) / (span + 1e-12)


def _signed_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    max_abs = float(np.nanmax(np.abs(values))) if len(values) else 0.0
    if not np.isfinite(max_abs) or max_abs <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return values / (max_abs + 1e-12)
