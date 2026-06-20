"""Replay a trained NQSE circuit on a real IBM Quantum backend.

This is a real-hardware replay: surrogate training remains local, then learned
circuits are transpiled, sampled, and decoded from physical counts.

Setup (never commit a token):
  D:\\ProgramData\\py2\\python.exe -m pip install qiskit qiskit-ibm-runtime
  $env:QISKIT_IBM_TOKEN = "your IBM Quantum token"
  D:\\ProgramData\\py2\\python.exe run_ibm_qpu_experiment.py --samples 32 --shots 2048
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def build_circuit(x: np.ndarray, theta: np.ndarray, n_qubits: int = 4, n_layers: int = 3):
    """Build the circular-entanglement circuit matching the local replay."""
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(n_qubits, n_qubits)
    for wire in range(n_qubits):
        value = float(x[wire % len(x)])
        circuit.ry(value, wire)
        circuit.rz(value * value / np.pi, wire)
    index = 0
    for layer in range(n_layers):
        for wire in range(n_qubits):
            value = float(x[wire % len(x)]) / np.sqrt(layer + 2.0)
            circuit.ry(value, wire)
            circuit.rz(value * value / np.pi, wire)
        for wire in range(n_qubits):
            circuit.rx(float(theta[index]), wire)
            circuit.ry(float(theta[index + 1]), wire)
            circuit.rz(float(theta[index + 2]), wire)
            index += 3
        for wire in range(n_qubits):
            circuit.cx(wire, (wire + 1) % n_qubits)
    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def features_from_counts(counts: dict[str, int], n_qubits: int = 4) -> np.ndarray:
    """Compute Z and neighboring ZZ expectations from computational counts."""
    total = sum(counts.values())
    if total == 0:
        raise ValueError("empty hardware counts")
    z = np.zeros(n_qubits)
    zz = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        # Qiskit prints c[n-1]..c[0]; measurement maps q[i] to c[i].
        bits = [int(bit) for bit in bitstring.replace(" ", "")[::-1]][:n_qubits]
        values = np.array([1.0 if bit == 0 else -1.0 for bit in bits])
        z += count * values
        zz += count * values * np.roll(values, -1)
    return np.concatenate([z, zz]) / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, help="backend name; default chooses least busy")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--shots", type=int, default=2048)
    args = parser.parse_args()

    token = os.environ.get("QISKIT_IBM_TOKEN")
    if not token:
        raise RuntimeError("Set QISKIT_IBM_TOKEN before submitting a real-QPU experiment.")
    try:
        from qiskit import transpile
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as error:
        raise RuntimeError("Install qiskit and qiskit-ibm-runtime in D:\\ProgramData\\py2.") from error

    # Delay local project imports so a missing token fails immediately.
    from qlea.attack import GeneralQuantumExtractor
    from qlea.torch_victims import ImageCNN, make_shape_images, train_torch_victim

    data = make_shape_images(n_samples=3000, image_size=16)
    victim = train_torch_victim("cnn_shape_images_ibm_qpu", ImageCNN(n_classes=4), data, epochs=4)
    projector = make_pipeline(StandardScaler(), PCA(n_components=8, random_state=7), StandardScaler())
    query_x = projector.fit_transform(data.x_train[:384].reshape(384, -1))
    eval_x = projector.transform(data.x_test[: args.samples].reshape(args.samples, -1))
    query_y = victim.predict(data.x_train[:384])
    victim_y = victim.predict(data.x_test[: args.samples])
    extractor = GeneralQuantumExtractor(n_qubits=4, n_layers=3, noise_kind="phase_flip", noise_p=0.002, seed=31)
    qnn = extractor.fit_from_labels(query_x, query_y, n_classes=4, epochs=10)["qnn"]
    analytic_pred = qnn.predict(eval_x)

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.backend(args.backend) if args.backend else service.least_busy(operational=True, simulator=False, min_num_qubits=4)
    if backend.num_qubits < 4:
        raise ValueError("selected backend has fewer than four qubits")
    circuits = [build_circuit(row, qnn.theta) for row in eval_x]
    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpile(circuits, backend=backend, optimization_level=1), shots=args.shots)
    print(f"submitted job {job.job_id()} to {backend.name}")
    result = job.result()
    features = np.vstack([features_from_counts(item.data.meas.get_counts()) for item in result])
    hardware_pred = np.argmax(features @ qnn.readout + qnn.bias, axis=1)
    row = {
        "backend": backend.name,
        "job_id": job.job_id(),
        "shots": args.shots,
        "samples": args.samples,
        "analytic_agreement": float(np.mean(analytic_pred == victim_y)),
        "hardware_agreement": float(np.mean(hardware_pred == victim_y)),
        "hardware_vs_analytic": float(np.mean(hardware_pred == analytic_pred)),
    }
    Path("outputs").mkdir(exist_ok=True)
    with Path("outputs/ibm_qpu_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
