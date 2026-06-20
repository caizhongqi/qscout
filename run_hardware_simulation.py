"""Hardware-style quantum simulation for the QNN extraction attack.

This script trains the QNN attack with the fast NumPy density-matrix simulator,
then replays the learned quantum circuit on PennyLane `default.mixed` with
finite shots and noise. It approximates how the attack behaves when the quantum
feature extractor is executed on a noisy quantum processor.

Run:
    & "D:\\ProgramData\\py2\\python.exe" run_hardware_simulation.py
"""

from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qlea.attack import GeneralQuantumExtractor
from qlea.pennylane_qnn import PennyLaneHardwareFeatureMap
from qlea.torch_victims import ImageCNN, make_shape_images, train_torch_victim


OUT_DIR = Path("outputs")


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = make_shape_images(n_samples=3000, image_size=16)
    victim = train_torch_victim(
        "cnn_shape_images_hardware_sim",
        ImageCNN(n_classes=4),
        data,
        epochs=4,
    )

    query_budget = 384
    eval_budget = 400
    projector = make_pipeline(
        StandardScaler(),
        PCA(n_components=8, random_state=7),
        StandardScaler(),
    )
    q_query = projector.fit_transform(flatten(data.x_train[:query_budget]))
    q_eval = projector.transform(flatten(data.x_test[:eval_budget]))
    query_labels = victim.predict(data.x_train[:query_budget])
    victim_eval = victim.predict(data.x_test[:eval_budget])

    extractor = GeneralQuantumExtractor(
        n_qubits=4,
        n_layers=3,
        noise_kind="phase_flip",
        noise_p=0.002,
        seed=31,
    )
    result = extractor.fit_from_labels(
        q_query,
        query_labels,
        n_classes=4,
        epochs=10,
    )
    qnn = result["qnn"]
    simulator_pred = qnn.predict(q_eval)

    rows = []
    for shots, noise_kind, noise_p in [
        (None, "none", 0.0),
        (2048, "phase_flip", 0.002),
        (1024, "phase_flip", 0.01),
        (1024, "amplitude_damping", 0.01),
        (512, "depolarizing", 0.005),
    ]:
        hardware_map = PennyLaneHardwareFeatureMap(
            n_qubits=4,
            n_layers=3,
            shots=shots,
            noise_kind=noise_kind,
            noise_p=noise_p,
            entanglement="circular",
        )
        features = hardware_map.transform(q_eval, qnn.theta)
        logits = features @ qnn.readout + qnn.bias
        hardware_pred = np.argmax(logits, axis=1)
        rows.append(
            {
                "shots": "analytic" if shots is None else shots,
                "noise_kind": noise_kind,
                "noise_p": noise_p,
                "victim_accuracy": float(np.mean(victim_eval == data.y_test[:eval_budget])),
                "simulator_agreement": float(np.mean(simulator_pred == victim_eval)),
                "hardware_agreement": float(np.mean(hardware_pred == victim_eval)),
                "hardware_vs_simulator": float(np.mean(hardware_pred == simulator_pred)),
            }
        )
        print(rows[-1])

    csv_path = OUT_DIR / "hardware_simulation_results.csv"
    header = [
        "shots",
        "noise_kind",
        "noise_p",
        "victim_accuracy",
        "simulator_agreement",
        "hardware_agreement",
        "hardware_vs_simulator",
    ]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in header) + "\n")
    print(f"results: {csv_path}")


if __name__ == "__main__":
    main()
