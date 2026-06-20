"""PennyLane hardware-like QNN feature map.

This module mirrors the in-house VQC in qlea.qnn, but executes it on PennyLane's
mixed-state device with finite shots and explicit noise channels. It is intended
for hardware-style validation after training the fast NumPy simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml


@dataclass
class PennyLaneHardwareFeatureMap:
    n_qubits: int = 4
    n_layers: int = 3
    shots: int | None = 1024
    noise_kind: str = "phase_flip"
    noise_p: float = 0.002
    entanglement: str = "circular"
    data_reuploading: bool = True

    @property
    def n_features(self) -> int:
        return self.n_qubits * 2

    def _device(self):
        return qml.device("default.mixed", wires=self.n_qubits, shots=self.shots)

    def _apply_noise(self) -> None:
        if self.noise_p <= 0.0 or self.noise_kind == "none":
            return
        for wire in range(self.n_qubits):
            if self.noise_kind == "phase_flip":
                qml.PhaseFlip(self.noise_p, wires=wire)
            elif self.noise_kind == "bit_flip":
                qml.BitFlip(self.noise_p, wires=wire)
            elif self.noise_kind == "amplitude_damping":
                qml.AmplitudeDamping(self.noise_p, wires=wire)
            elif self.noise_kind == "depolarizing":
                qml.DepolarizingChannel(self.noise_p, wires=wire)
            else:
                raise ValueError(f"Unsupported PennyLane noise: {self.noise_kind}")

    def _encode(self, x: np.ndarray, layer: int) -> None:
        scale = 1.0 / np.sqrt(layer + 1.0)
        for wire in range(self.n_qubits):
            angle = float(x[wire % x.shape[0]]) * scale
            qml.RY(angle, wires=wire)
            qml.RZ(angle * angle / np.pi, wires=wire)

    def _entangle(self) -> None:
        if self.entanglement == "none":
            return
        if self.entanglement == "linear":
            pairs = [(wire, wire + 1) for wire in range(self.n_qubits - 1)]
        elif self.entanglement == "circular":
            pairs = [(wire, (wire + 1) % self.n_qubits) for wire in range(self.n_qubits)]
        elif self.entanglement == "star":
            pairs = [(0, wire) for wire in range(1, self.n_qubits)]
        elif self.entanglement == "full":
            pairs = [
                (control, target)
                for control in range(self.n_qubits)
                for target in range(control + 1, self.n_qubits)
            ]
        else:
            raise ValueError(f"Unsupported entanglement pattern: {self.entanglement}")
        for control, target in pairs:
            qml.CNOT(wires=[control, target])

    def transform_one(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        dev = self._device()

        @qml.qnode(dev)
        def circuit(x_value, theta_value):
            self._encode(x_value, layer=0)
            self._apply_noise()
            idx = 0
            for layer in range(self.n_layers):
                if self.data_reuploading:
                    self._encode(x_value, layer=layer + 1)
                for wire in range(self.n_qubits):
                    qml.RX(theta_value[idx], wires=wire)
                    idx += 1
                    qml.RY(theta_value[idx], wires=wire)
                    idx += 1
                    qml.RZ(theta_value[idx], wires=wire)
                    idx += 1
                self._entangle()
                self._apply_noise()
            observables = [qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits)]
            observables.extend(
                [
                    qml.expval(qml.PauliZ(wire) @ qml.PauliZ((wire + 1) % self.n_qubits))
                    for wire in range(self.n_qubits)
                ]
            )
            return observables

        return np.array(circuit(x, theta), dtype=float)

    def transform(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.vstack([self.transform_one(row, theta) for row in x])
