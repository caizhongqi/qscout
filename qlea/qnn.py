"""Variational quantum feature model trained from hard labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quantum import (
    apply_kraus,
    apply_unitary,
    cnot_operator,
    noise_kraus,
    rx,
    ry,
    rz,
    zero_density,
    z_expectation,
    zz_expectation,
    one_qubit_operator,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class QuantumFeatureMap:
    n_qubits: int = 4
    n_layers: int = 2
    noise_kind: str = "none"
    noise_p: float = 0.0
    entanglement: str = "circular"
    data_reuploading: bool = True
    measure_zz: bool = True
    feature_cycling: bool = True

    @property
    def n_parameters(self) -> int:
        return self.n_layers * self.n_qubits * 3

    @property
    def n_features(self) -> int:
        return self.n_qubits * (2 if self.measure_zz else 1)

    def initial_parameters(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 0.15, size=self.n_parameters)

    def _apply_noise_all(self, rho: np.ndarray) -> np.ndarray:
        if self.noise_p <= 0.0 or self.noise_kind == "none":
            return rho
        kraus = noise_kraus(self.noise_kind, self.noise_p)
        for wire in range(self.n_qubits):
            rho = apply_kraus(rho, kraus, wire, self.n_qubits)
        return rho

    def _entangle(self, rho: np.ndarray) -> np.ndarray:
        if self.entanglement == "none":
            return rho
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
            rho = apply_unitary(rho, cnot_operator(control, target, self.n_qubits))
        return rho

    def _encode(self, rho: np.ndarray, x: np.ndarray, layer: int = 0) -> np.ndarray:
        scale = 1.0 / np.sqrt(layer + 1.0)
        for wire in range(self.n_qubits):
            feature_index = wire
            if self.feature_cycling:
                # Each reuploading layer sees a different chunk of the public
                # compressed representation before cycling back to the start.
                feature_index = wire + layer * self.n_qubits
            angle = float(x[feature_index % x.shape[0]]) * scale
            rho = apply_unitary(rho, one_qubit_operator(ry(angle), wire, self.n_qubits))
            rho = apply_unitary(
                rho,
                one_qubit_operator(rz(angle * angle / np.pi), wire, self.n_qubits),
            )
        return rho

    def transform_one(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        rho = zero_density(self.n_qubits)
        rho = self._encode(rho, x, layer=0)
        rho = self._apply_noise_all(rho)

        idx = 0
        for layer in range(self.n_layers):
            if self.data_reuploading:
                rho = self._encode(rho, x, layer=layer + 1)
            for wire in range(self.n_qubits):
                rho = apply_unitary(
                    rho, one_qubit_operator(rx(theta[idx]), wire, self.n_qubits)
                )
                idx += 1
                rho = apply_unitary(
                    rho, one_qubit_operator(ry(theta[idx]), wire, self.n_qubits)
                )
                idx += 1
                rho = apply_unitary(
                    rho, one_qubit_operator(rz(theta[idx]), wire, self.n_qubits)
                )
                idx += 1
            rho = self._entangle(rho)
            rho = self._apply_noise_all(rho)

        z_terms = [z_expectation(rho, wire, self.n_qubits) for wire in range(self.n_qubits)]
        if not self.measure_zz:
            return np.array(z_terms, dtype=float)
        zz_terms = [
            zz_expectation(rho, wire, (wire + 1) % self.n_qubits, self.n_qubits)
            for wire in range(self.n_qubits)
        ]
        return np.array(z_terms + zz_terms, dtype=float)

    def transform(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.vstack([self.transform_one(row, theta) for row in x])


@dataclass
class QuantumClassifier:
    feature_map: QuantumFeatureMap
    n_classes: int
    theta: np.ndarray
    readout: np.ndarray
    bias: np.ndarray

    @classmethod
    def create(
        cls,
        feature_map: QuantumFeatureMap,
        n_classes: int,
        rng: np.random.Generator,
    ) -> "QuantumClassifier":
        theta = feature_map.initial_parameters(rng)
        readout = rng.normal(0.0, 0.1, size=(feature_map.n_features, n_classes))
        bias = np.zeros(n_classes)
        return cls(feature_map, n_classes, theta, readout, bias)

    def logits(self, x: np.ndarray) -> np.ndarray:
        phi = self.feature_map.transform(x, self.theta)
        return phi @ self.readout + self.bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(x), axis=1)

    def loss(self, x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> float:
        probs = softmax(self.logits(x))
        n = x.shape[0]
        ce = -np.log(probs[np.arange(n), y] + 1e-12).mean()
        return float(ce + l2 * np.mean(self.readout**2))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 35,
        batch_size: int = 64,
        lr_readout: float = 0.15,
        lr_theta: float = 0.03,
        spsa_delta: float = 0.08,
        seed: int = 11,
    ) -> list[float]:
        rng = np.random.default_rng(seed)
        history: list[float] = []
        for epoch in range(epochs):
            order = rng.permutation(x.shape[0])
            for start in range(0, x.shape[0], batch_size):
                batch = order[start : start + batch_size]
                xb = x[batch]
                yb = y[batch]
                phi = self.feature_map.transform(xb, self.theta)
                probs = softmax(phi @ self.readout + self.bias)
                probs[np.arange(yb.shape[0]), yb] -= 1.0
                grad_logits = probs / yb.shape[0]
                self.readout -= lr_readout * (phi.T @ grad_logits + 1e-3 * self.readout)
                self.bias -= lr_readout * grad_logits.sum(axis=0)

            direction = rng.choice([-1.0, 1.0], size=self.theta.shape)
            theta_base = self.theta.copy()
            self.theta = theta_base + spsa_delta * direction
            loss_plus = self.loss(x[: min(128, x.shape[0])], y[: min(128, y.shape[0])])
            self.theta = theta_base - spsa_delta * direction
            loss_minus = self.loss(x[: min(128, x.shape[0])], y[: min(128, y.shape[0])])
            grad_theta = ((loss_plus - loss_minus) / (2.0 * spsa_delta)) * direction
            self.theta = theta_base - lr_theta * grad_theta
            history.append(self.loss(x, y))
        return history
