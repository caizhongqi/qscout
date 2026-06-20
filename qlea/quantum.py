"""Small density-matrix quantum simulator for NISQ VQC experiments."""

from __future__ import annotations

import numpy as np


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=complex
    )


def kron_all(ops: list[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def one_qubit_operator(gate: np.ndarray, wire: int, n_qubits: int) -> np.ndarray:
    return kron_all([gate if i == wire else I2 for i in range(n_qubits)])


def cnot_operator(control: int, target: int, n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    op = np.zeros((dim, dim), dtype=complex)
    for basis in range(dim):
        bits = [(basis >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)]
        if bits[control]:
            bits[target] ^= 1
        out = 0
        for bit in bits:
            out = (out << 1) | bit
        op[out, basis] = 1
    return op


def zero_density(n_qubits: int) -> np.ndarray:
    psi = np.zeros((2**n_qubits, 1), dtype=complex)
    psi[0, 0] = 1.0
    return psi @ psi.conj().T


def apply_unitary(rho: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    return unitary @ rho @ unitary.conj().T


def apply_kraus(
    rho: np.ndarray, kraus_ops: list[np.ndarray], wire: int, n_qubits: int
) -> np.ndarray:
    out = np.zeros_like(rho)
    for k in kraus_ops:
        full = one_qubit_operator(k, wire, n_qubits)
        out += full @ rho @ full.conj().T
    return out


def noise_kraus(kind: str, p: float) -> list[np.ndarray]:
    p = float(np.clip(p, 0.0, 1.0))
    if p == 0.0 or kind == "none":
        return [I2]
    if kind == "phase_flip":
        return [np.sqrt(1.0 - p) * I2, np.sqrt(p) * Z]
    if kind == "bit_flip":
        return [np.sqrt(1.0 - p) * I2, np.sqrt(p) * X]
    if kind == "amplitude_damping":
        return [
            np.array([[1, 0], [0, np.sqrt(1.0 - p)]], dtype=complex),
            np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex),
        ]
    raise ValueError(f"Unsupported noise channel: {kind}")


def z_expectation(rho: np.ndarray, wire: int, n_qubits: int) -> float:
    observable = one_qubit_operator(Z, wire, n_qubits)
    return float(np.real(np.trace(rho @ observable)))


def zz_expectation(rho: np.ndarray, left: int, right: int, n_qubits: int) -> float:
    ops = [I2 for _ in range(n_qubits)]
    ops[left] = Z
    ops[right] = Z
    observable = kron_all(ops)
    return float(np.real(np.trace(rho @ observable)))
