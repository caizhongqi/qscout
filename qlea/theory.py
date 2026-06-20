"""Conservative, checkable theory utilities for the NQSE prototype.

The bounds in this module are deliberately sufficient rather than tight.  They
turn the assumptions made in the paper (local CPTP noise and finite-shot Pauli
measurements) into quantities that can be reported next to an experiment.
"""

from __future__ import annotations

import math

import numpy as np


def hard_label_information_lower_bound(
    n_unknown_real_parameters: int, n_classes: int, bits_per_parameter: int = 1
) -> int:
    """Minimum queries needed to identify a discretised parameter vector.

    A hard-label response has at most log2(C) bits.  This is an information
    lower bound, not an achievable extraction algorithm.
    """
    if n_unknown_real_parameters < 1 or n_classes < 2 or bits_per_parameter < 1:
        raise ValueError("parameters must be positive and n_classes must be at least two")
    return math.ceil(
        n_unknown_real_parameters * bits_per_parameter / math.log2(n_classes)
    )


def finite_class_disagreement_bound(
    n_queries: int, log_hypothesis_count: float, delta: float = 0.05
) -> float:
    """Uniform-convergence term for a finite, quantised surrogate class.

    If log_hypothesis_count is log(|H|), then with probability >= 1-delta the
    population disagreement differs from empirical disagreement by at most the
    returned value.  A continuous QNN must be quantised or replaced by a
    covering-number/pseudodimension bound before this statement is used.
    """
    if n_queries < 1 or not 0.0 < delta < 1.0:
        raise ValueError("n_queries must be positive and delta must be in (0, 1)")
    return math.sqrt((log_hypothesis_count + math.log(2.0 / delta)) / (2.0 * n_queries))


def local_noise_feature_bound(n_noise_sites: int, noise_probability: float) -> float:
    """Bound Pauli-observable drift under independent local CPTP noise.

    For a local channel whose diamond distance from identity is at most 2p,
    telescoping composition gives |E[O]_noisy-E[O]_ideal| <= min(2, 2 G p)
    for any observable O with operator norm at most one and G noise sites.
    """
    if n_noise_sites < 0 or not 0.0 <= noise_probability <= 1.0:
        raise ValueError("invalid noise configuration")
    return min(2.0, 2.0 * n_noise_sites * noise_probability)


def shot_feature_bound(n_observables: int, shots: int, delta: float = 0.05) -> float:
    """Simultaneous Hoeffding bound for Pauli expectations in [-1, 1]."""
    if n_observables < 1 or shots < 1 or not 0.0 < delta < 1.0:
        raise ValueError("invalid shot-bound configuration")
    return math.sqrt(math.log(2.0 * n_observables / delta) / (2.0 * shots))


def logit_perturbation_bound(
    readout: np.ndarray, feature_error: float
) -> float:
    """Upper bound on the largest readout-logit perturbation."""
    return float(np.linalg.norm(readout, ord=2) * math.sqrt(readout.shape[0]) * feature_error)


def certified_prediction_mask(
    logits: np.ndarray, logit_error: float
) -> np.ndarray:
    """Return samples whose argmax cannot change under the sufficient bound."""
    ordered = np.sort(logits, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    return margins > 2.0 * logit_error
