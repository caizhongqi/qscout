"""Quantum-query primitives for the critical sparse-network regime.

The fast detector implemented here is intentionally named an *isolation*
detector, not a general quantum perfect-matching algorithm.  Near the random
bipartite perfect-matching threshold ``p=(log d + c)/d``, isolated vertices are
the asymptotically dominant obstruction.  Detecting an isolated row/column can
be organized as nested Grover search and costs O(d) adjacency-oracle queries.
Away from that regime, Hall obstructions with no isolated vertex can occur, so
an exact perfect-matching routine is still required for a worst-case decision.

This distinction is important for the paper's complexity claim:

* general structural injectivity: use exact classical max-flow or a previously
  known generic quantum matching algorithm;
* critical random regime: the isolation witness is asymptotically sufficient
  in probability and admits the O(d) nested-search query bound below.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GroverEstimate:
    search_size: int
    marked_items: int
    iterations: int
    success_probability: float


@dataclass(frozen=True)
class CriticalIsolationResources:
    dimension: int
    inner_grover_iterations: int
    outer_grover_iterations: int
    adjacency_oracle_queries_raw: int
    adjacency_queries_over_d: float
    logical_qubits_estimate: int


def optimal_grover_iterations(search_size: int, marked_items: int = 1) -> int:
    """Return the nearest standard Grover iteration count for known M marked items."""

    if search_size <= 0:
        raise ValueError("search_size must be positive")
    if not 0 < marked_items <= search_size:
        raise ValueError("marked_items must lie in [1, search_size]")
    theta = math.asin(math.sqrt(marked_items / search_size))
    return max(0, int(round(math.pi / (4.0 * theta) - 0.5)))


def grover_success_probability(
    search_size: int,
    marked_items: int = 1,
    iterations: int | None = None,
) -> GroverEstimate:
    """Exact ideal-state success probability of textbook amplitude amplification."""

    if iterations is None:
        iterations = optimal_grover_iterations(search_size, marked_items)
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    theta = math.asin(math.sqrt(marked_items / search_size))
    probability = math.sin((2 * iterations + 1) * theta) ** 2
    return GroverEstimate(
        search_size=search_size,
        marked_items=marked_items,
        iterations=iterations,
        success_probability=float(probability),
    )


def critical_isolation_resource_estimate(dimension: int) -> CriticalIsolationResources:
    """Estimate nested-Grover adjacency queries for one isolated-side witness.

    The accounting uses four adjacency-oracle phase/check calls per inner/outer
    iteration pair.  It is a transparent raw-query proxy rather than a claim
    about a particular fault-tolerant decomposition.  Since both iteration
    counts scale as Theta(sqrt(d)), the product scales as Theta(d).
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    inner = optimal_grover_iterations(dimension, 1)
    outer = optimal_grover_iterations(dimension, 1)
    raw_queries = 4 * max(1, inner) * max(1, outer)
    logd = max(1, int(math.ceil(math.log2(dimension))))
    # row index + column index + work/flag register, each O(log d)
    qubits = 3 * logd + 3
    return CriticalIsolationResources(
        dimension=dimension,
        inner_grover_iterations=inner,
        outer_grover_iterations=outer,
        adjacency_oracle_queries_raw=raw_queries,
        adjacency_queries_over_d=float(raw_queries / dimension),
        logical_qubits_estimate=qubits,
    )


def classical_isolated_vertices(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact isolated row and column indices of a bipartite adjacency matrix."""

    matrix = np.asarray(adjacency, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError("adjacency must be a two-dimensional matrix")
    isolated_rows = np.flatnonzero(np.count_nonzero(matrix, axis=1) == 0)
    isolated_columns = np.flatnonzero(np.count_nonzero(matrix, axis=0) == 0)
    return isolated_rows, isolated_columns


def has_isolation_obstruction(adjacency: np.ndarray) -> bool:
    """Whether a bipartite graph has an isolated vertex on either side."""

    rows, columns = classical_isolated_vertices(adjacency)
    return bool(rows.size or columns.size)


def critical_edge_probability(dimension: int, c: float = 0.0) -> float:
    """Return ``p=(log d + c)/d``, clipped to [0, 1]."""

    if dimension <= 1:
        raise ValueError("dimension must be greater than one")
    return float(np.clip((math.log(dimension) + c) / dimension, 0.0, 1.0))


def asymptotic_perfect_matching_probability(c: float) -> float:
    """Critical-window limit ``exp(-2 exp(-c))`` for G_{d,d,p}.

    At ``p=(log d+c)/d``, the probability of a perfect matching has the same
    limiting law as the disappearance of isolated vertices.  This is a known
    random-graph threshold result; this function is included only as the theory
    curve used by the neural-collision experiments.
    """

    return float(math.exp(-2.0 * math.exp(-float(c))))
