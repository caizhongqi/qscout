from __future__ import annotations

import math

import numpy as np
import pytest

from qlea.neural_collision import (
    asymptotic_perfect_matching_probability,
    classical_isolated_vertices,
    critical_edge_probability,
    critical_isolation_resource_estimate,
    grover_success_probability,
    maximum_bipartite_matching_size,
)


def test_hopcroft_karp_matching_cardinality() -> None:
    perfect = np.eye(4, dtype=bool)
    deficient = perfect.copy()
    deficient[:, 3] = False

    assert maximum_bipartite_matching_size(perfect) == 4
    assert maximum_bipartite_matching_size(deficient) == 3


def test_isolation_detection_on_both_sides() -> None:
    matrix = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    )
    rows, columns = classical_isolated_vertices(matrix)

    assert rows.tolist() == []
    assert columns.tolist() == [2]


def test_critical_probability_and_limit_curve() -> None:
    d = 128
    p = critical_edge_probability(d, c=0.0)
    assert p == pytest.approx(math.log(d) / d)
    assert asymptotic_perfect_matching_probability(0.0) == pytest.approx(math.exp(-2.0))


def test_grover_demo_matches_known_small_dimension_values() -> None:
    estimate = grover_success_probability(8)
    assert estimate.iterations == 2
    assert estimate.success_probability == pytest.approx(0.9453125)


def test_nested_query_proxy_is_linear_order() -> None:
    small = critical_isolation_resource_estimate(64)
    large = critical_isolation_resource_estimate(256)

    assert small.inner_grover_iterations == 6
    assert large.inner_grover_iterations == 12
    # d grows by 4x; the raw nested-Grover query proxy also grows by 4x.
    assert large.adjacency_oracle_queries_raw == 4 * small.adjacency_oracle_queries_raw
