from __future__ import annotations

import pytest


pytest.importorskip("qiskit")

from qlea.neural_collision.qiskit_grover import statevector_marked_probability  # noqa: E402
from qlea.neural_collision.quantum import grover_success_probability  # noqa: E402


def test_qiskit_statevector_matches_analytic_grover_probability() -> None:
    num_qubits = 3
    search_size = 2**num_qubits
    analytic = grover_success_probability(search_size)
    simulated = statevector_marked_probability(num_qubits, marked_index=5)
    assert simulated == pytest.approx(analytic.success_probability, abs=1e-10)
