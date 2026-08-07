"""Optional Qiskit circuits for the Grover primitive used by the detector.

This file validates the amplitude-amplification building block on an explicit
statevector circuit.  It does *not* hide the oracle cost: a full coherent
isolation detector additionally needs a reversible adjacency oracle and a
reversible row-isolation predicate.  The query-complexity accounting for that
nested construction lives in :mod:`qlea.neural_collision.quantum`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .quantum import optimal_grover_iterations


def _qiskit():
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Qiskit is required for qlea.neural_collision.qiskit_grover; "
            "install qscout[qpu] or qscout[all]"
        ) from exc
    return QuantumCircuit, Statevector


def _phase_flip_basis_state(circuit: Any, qubits: list[int], marked_index: int) -> None:
    """Apply a phase -1 to one computational-basis state."""

    n = len(qubits)
    if n <= 0:
        raise ValueError("at least one qubit is required")
    if not 0 <= marked_index < 2**n:
        raise ValueError("marked_index is outside the basis-state range")

    # Qiskit uses little-endian basis indexing: qubit 0 is the least
    # significant bit.  X gates map zero-controls to one-controls.
    zero_qubits = [q for bit, q in enumerate(qubits) if ((marked_index >> bit) & 1) == 0]
    for qubit in zero_qubits:
        circuit.x(qubit)

    if n == 1:
        circuit.z(qubits[0])
    else:
        target = qubits[-1]
        controls = qubits[:-1]
        circuit.h(target)
        circuit.mcx(controls, target)
        circuit.h(target)

    for qubit in reversed(zero_qubits):
        circuit.x(qubit)


def _diffuser(circuit: Any, qubits: list[int]) -> None:
    for qubit in qubits:
        circuit.h(qubit)
        circuit.x(qubit)

    if len(qubits) == 1:
        circuit.z(qubits[0])
    else:
        target = qubits[-1]
        controls = qubits[:-1]
        circuit.h(target)
        circuit.mcx(controls, target)
        circuit.h(target)

    for qubit in qubits:
        circuit.x(qubit)
        circuit.h(qubit)


def build_unique_marked_grover_circuit(
    num_qubits: int,
    marked_index: int,
    *,
    iterations: int | None = None,
    measure: bool = False,
):
    """Build a textbook Grover circuit with exactly one marked basis state."""

    QuantumCircuit, _ = _qiskit()
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    search_size = 2**num_qubits
    if not 0 <= marked_index < search_size:
        raise ValueError("marked_index must lie in [0, 2**num_qubits)")
    if iterations is None:
        iterations = optimal_grover_iterations(search_size, 1)
    if iterations < 0:
        raise ValueError("iterations must be non-negative")

    circuit = QuantumCircuit(num_qubits, num_qubits if measure else 0)
    qubits = list(range(num_qubits))
    circuit.h(qubits)
    for _ in range(iterations):
        _phase_flip_basis_state(circuit, qubits, marked_index)
        _diffuser(circuit, qubits)
    if measure:
        circuit.measure(qubits, qubits)
    return circuit


def statevector_marked_probability(
    num_qubits: int,
    marked_index: int,
    *,
    iterations: int | None = None,
) -> float:
    """Simulate the ideal circuit and return marked-state probability."""

    _, Statevector = _qiskit()
    circuit = build_unique_marked_grover_circuit(
        num_qubits,
        marked_index,
        iterations=iterations,
        measure=False,
    )
    state = Statevector.from_instruction(circuit)
    probabilities = np.asarray(state.probabilities(), dtype=float)
    return float(probabilities[marked_index])
