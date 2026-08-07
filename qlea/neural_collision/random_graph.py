"""Random-graph experiments for structural neural collision transitions.

The sparse layered model uses independent ``G_{d,d,p}`` supports between
consecutive width-``d`` layers.  Full structural column rank requires ``d``
vertex-disjoint input-output paths.  In the equal-width layered model, this is
equivalent to every consecutive bipartite support admitting a perfect matching.

The critical window is parameterized as

    p = (log d + c) / d.

For one bipartite layer the known limiting perfect-matching probability is
``exp(-2 exp(-c))``; for ``L`` independent transitions it becomes
``exp(-2 L exp(-c))``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .quantum import critical_edge_probability
from .structure import structural_path_rank


Array = np.ndarray


@dataclass(frozen=True)
class PhaseTransitionPoint:
    dimension: int
    layers: int
    c: float
    edge_probability: float
    repetitions: int
    injective_probability: float
    noninjective_probability: float
    residual_failure_rate: float
    isolated_failure_share: float
    theory_probability: float
    mean_path_rank: float
    mean_cdr: float


def maximum_bipartite_matching_size(adjacency: Array) -> int:
    """Hopcroft-Karp maximum matching cardinality for a Boolean matrix.

    Rows are the left partition and columns are the right partition.  The
    orientation is irrelevant for cardinality and is kept separate from the
    destination-by-source matrix convention used in ``structure.py``.
    """

    matrix = np.asarray(adjacency, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError("adjacency must be two-dimensional")
    n_left, n_right = matrix.shape
    neighbors = [np.flatnonzero(matrix[u]).tolist() for u in range(n_left)]

    pair_left = [-1] * n_left
    pair_right = [-1] * n_right
    distance = [0] * n_left
    infinity = n_left + n_right + 1

    def bfs() -> bool:
        queue: deque[int] = deque()
        found = False
        for u in range(n_left):
            if pair_left[u] == -1:
                distance[u] = 0
                queue.append(u)
            else:
                distance[u] = infinity
        while queue:
            u = queue.popleft()
            for v in neighbors[u]:
                matched = pair_right[v]
                if matched == -1:
                    found = True
                elif distance[matched] == infinity:
                    distance[matched] = distance[u] + 1
                    queue.append(matched)
        return found

    def dfs(u: int) -> bool:
        for v in neighbors[u]:
            matched = pair_right[v]
            if matched == -1 or (
                distance[matched] == distance[u] + 1 and dfs(matched)
            ):
                pair_left[u] = v
                pair_right[v] = u
                return True
        distance[u] = infinity
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_left[u] == -1 and dfs(u):
                matching += 1
    return matching


def has_isolated_vertex(adjacency: Array) -> bool:
    matrix = np.asarray(adjacency, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError("adjacency must be two-dimensional")
    return bool(
        np.any(np.count_nonzero(matrix, axis=0) == 0)
        or np.any(np.count_nonzero(matrix, axis=1) == 0)
    )


def sample_layered_supports(
    dimension: int,
    layers: int,
    edge_probability: float,
    rng: np.random.Generator,
) -> tuple[Array, ...]:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if layers <= 0:
        raise ValueError("layers must be positive")
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("edge_probability must lie in [0, 1]")
    return tuple(
        rng.random((dimension, dimension)) < edge_probability
        for _ in range(layers)
    )


def _full_path_rank_fast(supports: tuple[Array, ...]) -> bool:
    dimension = supports[0].shape[0]
    return all(maximum_bipartite_matching_size(matrix) == dimension for matrix in supports)


def estimate_phase_transition_point(
    dimension: int,
    layers: int,
    c: float,
    repetitions: int,
    *,
    seed: int = 0,
    compute_mean_path_rank: bool = True,
) -> PhaseTransitionPoint:
    """Monte-Carlo estimate at one point of the matching threshold window."""

    if dimension <= 1:
        raise ValueError("dimension must exceed one")
    if layers <= 0:
        raise ValueError("layers must be positive")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    p = critical_edge_probability(dimension, c)
    rng = np.random.default_rng(seed)

    injective = 0
    isolated_failures = 0
    residual_failures = 0
    rank_sum = 0.0

    for _ in range(repetitions):
        supports = sample_layered_supports(dimension, layers, p, rng)
        full_rank = _full_path_rank_fast(supports)
        isolation = any(has_isolated_vertex(matrix) for matrix in supports)
        if full_rank:
            injective += 1
        elif isolation:
            isolated_failures += 1
        else:
            residual_failures += 1

        if compute_mean_path_rank:
            rank_sum += structural_path_rank(supports)

    failures = repetitions - injective
    pinj = injective / repetitions
    noninj = failures / repetitions
    residual_rate = residual_failures / repetitions
    iso_share = isolated_failures / failures if failures else 0.0
    if compute_mean_path_rank:
        mean_rank = rank_sum / repetitions
        mean_cdr = (dimension - mean_rank) / dimension
    else:
        mean_rank = float("nan")
        mean_cdr = float("nan")

    theory = math.exp(-2.0 * layers * math.exp(-float(c)))

    return PhaseTransitionPoint(
        dimension=dimension,
        layers=layers,
        c=float(c),
        edge_probability=float(p),
        repetitions=repetitions,
        injective_probability=float(pinj),
        noninjective_probability=float(noninj),
        residual_failure_rate=float(residual_rate),
        isolated_failure_share=float(iso_share),
        theory_probability=float(theory),
        mean_path_rank=float(mean_rank),
        mean_cdr=float(mean_cdr),
    )


def phase_transition_sweep(
    dimensions: Iterable[int],
    c_values: Iterable[float],
    *,
    layers: int = 1,
    repetitions: int = 200,
    seed: int = 0,
    compute_mean_path_rank: bool = True,
) -> list[PhaseTransitionPoint]:
    """Run a deterministic seeded sweep over dimensions and critical offsets."""

    points: list[PhaseTransitionPoint] = []
    master = np.random.SeedSequence(seed)
    parameter_pairs = [(int(d), float(c)) for d in dimensions for c in c_values]
    children = master.spawn(len(parameter_pairs))
    for (dimension, c), child in zip(parameter_pairs, children):
        child_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        points.append(
            estimate_phase_transition_point(
                dimension,
                layers,
                c,
                repetitions,
                seed=child_seed,
                compute_mean_path_rank=compute_mean_path_rank,
            )
        )
    return points
