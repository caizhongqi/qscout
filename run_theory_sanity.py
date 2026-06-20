"""Compute theory quantities for the QNN circuit used in the hardware replay.

Run:
    & "D:\\ProgramData\\py2\\python.exe" run_theory_sanity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qlea.theory import (
    certified_prediction_mask,
    finite_class_disagreement_bound,
    hard_label_information_lower_bound,
    local_noise_feature_bound,
    logit_perturbation_bound,
    shot_feature_bound,
)


def main() -> None:
    # Matches the 4-qubit, 3-layer Z/ZZ circuit used by hardware simulation.
    n_qubits, n_layers, n_classes, noise_p, shots = 4, 3, 4, 0.01, 1024
    n_features = 2 * n_qubits
    n_noise_sites = n_qubits * (n_layers + 1)
    feature_noise = local_noise_feature_bound(n_noise_sites, noise_p)
    feature_shots = shot_feature_bound(n_features, shots)

    # A fixed seed makes this only a mathematical sanity example; the actual
    # paper run should replace this matrix with a saved trained readout.
    readout = np.random.default_rng(31).normal(0.0, 0.1, size=(n_features, n_classes))
    logit_bound = logit_perturbation_bound(readout, feature_noise + feature_shots)
    demo_logits = np.array([[1.2, 0.1, -0.2, -0.5], [0.31, 0.30, -0.1, -0.2]])
    certified = certified_prediction_mask(demo_logits, logit_bound)

    report = {
        "hard_label_lower_bound_queries": hard_label_information_lower_bound(
            n_unknown_real_parameters=64, n_classes=n_classes, bits_per_parameter=8
        ),
        "finite_class_generalization_term": finite_class_disagreement_bound(
            n_queries=1024, log_hypothesis_count=32.0
        ),
        "n_noise_sites": n_noise_sites,
        "local_noise_feature_bound": feature_noise,
        "finite_shot_feature_bound": feature_shots,
        "combined_logit_bound": logit_bound,
        "demo_certified_mask": certified.tolist(),
        "warning": "Bounds are conservative sufficient certificates, not evidence of quantum advantage.",
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/theory_sanity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
