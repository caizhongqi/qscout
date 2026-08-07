"""Structural path certificates for sparse layered neural maps.

For a layered directed acyclic graph with independently parameterized non-zero
edge weights, the generic rank of the input-to-output transfer matrix equals
the maximum number of vertex-disjoint input-to-output paths.  This module
computes that path number by a unit-capacity max-flow construction.

Important scope restriction
---------------------------
The generic-rank statement assumes independent edge parameters.  It therefore
applies directly to ordinary sparse fully connected layers, but not verbatim to
weight-sharing architectures such as convolutions.  For CNNs, use the numerical
Jacobian/linear-region certificate unless the lifted convolutional operator is
analyzed with its parameter-sharing constraints explicitly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class StructuralPathCertificate:
    """Graph-theoretic generic-rank certificate for a layered sparse map."""

    layer_widths: tuple[int, ...]
    path_rank: int
    input_dim: int
    output_dim: int
    path_nullity: int
    collision_deficiency_ratio: float
    active_vertices_per_layer: tuple[int, ...]
    first_layer_isolated_inputs: int
    last_layer_isolated_outputs: int

    @property
    def structurally_full_column_rank(self) -> bool:
        return self.path_rank == self.input_dim

    @property
    def structurally_collision_prone(self) -> bool:
        return self.path_rank < self.input_dim


class _Dinic:
    """Small integer-capacity Dinic implementation used to avoid dependencies."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.graph: list[list[list[int]]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        forward = [v, capacity, 0]
        backward = [u, 0, 0]
        forward.append(len(self.graph[v]))
        backward.append(len(self.graph[u]))
        self.graph[u].append(forward)
        self.graph[v].append(backward)

    def max_flow(self, source: int, sink: int) -> int:
        total = 0
        while True:
            level = [-1] * self.n
            level[source] = 0
            queue: deque[int] = deque([source])
            while queue:
                u = queue.popleft()
                for edge in self.graph[u]:
                    v, capacity, flow, _ = edge
                    if level[v] < 0 and flow < capacity:
                        level[v] = level[u] + 1
                        queue.append(v)
            if level[sink] < 0:
                return total

            it = [0] * self.n

            def dfs(u: int, pushed: int) -> int:
                if u == sink:
                    return pushed
                while it[u] < len(self.graph[u]):
                    edge = self.graph[u][it[u]]
                    v, capacity, flow, reverse_index = edge
                    if level[v] == level[u] + 1 and flow < capacity:
                        amount = dfs(v, min(pushed, capacity - flow))
                        if amount:
                            edge[2] += amount
                            self.graph[v][reverse_index][2] -= amount
                            return amount
                    it[u] += 1
                return 0

            while True:
                pushed = dfs(source, 10**9)
                if not pushed:
                    break
                total += pushed


def _validate_supports(
    supports: Sequence[Array],
    vertex_masks: Sequence[Array] | None,
) -> tuple[tuple[Array, ...], tuple[Array, ...], tuple[int, ...]]:
    if not supports:
        raise ValueError("supports must contain at least one inter-layer adjacency")

    support_tuple = tuple(np.asarray(matrix, dtype=bool) for matrix in supports)
    if any(matrix.ndim != 2 for matrix in support_tuple):
        raise ValueError("each support matrix must be two-dimensional")

    widths = [int(support_tuple[0].shape[1])]
    previous = widths[0]
    for index, matrix in enumerate(support_tuple):
        if matrix.shape[1] != previous:
            raise ValueError(
                f"supports[{index}] has source width {matrix.shape[1]}, expected {previous}"
            )
        previous = int(matrix.shape[0])
        widths.append(previous)

    if vertex_masks is None:
        mask_tuple = tuple(np.ones(width, dtype=bool) for width in widths)
    else:
        if len(vertex_masks) != len(widths):
            raise ValueError(
                "vertex_masks must contain one mask for every layer, including input and output"
            )
        masks: list[Array] = []
        for index, (mask, width) in enumerate(zip(vertex_masks, widths)):
            arr = np.asarray(mask, dtype=bool)
            if arr.ndim != 1 or arr.shape[0] != width:
                raise ValueError(
                    f"vertex_masks[{index}] must have shape ({width},), got {arr.shape}"
                )
            masks.append(arr)
        mask_tuple = tuple(masks)

    return support_tuple, mask_tuple, tuple(widths)


def structural_path_rank(
    supports: Sequence[Array],
    *,
    vertex_masks: Sequence[Array] | None = None,
) -> int:
    """Return the maximum number of vertex-disjoint input-output paths.

    ``supports[l]`` has shape ``(width[l+1], width[l])`` and uses the same
    destination-by-source convention as a dense layer weight matrix.
    """

    support_tuple, masks, widths = _validate_supports(supports, vertex_masks)

    # Vertex splitting: each active network vertex receives unit capacity.
    node_in: dict[tuple[int, int], int] = {}
    node_out: dict[tuple[int, int], int] = {}
    next_node = 0
    for layer, width in enumerate(widths):
        for index in range(width):
            if not masks[layer][index]:
                continue
            node_in[(layer, index)] = next_node
            next_node += 1
            node_out[(layer, index)] = next_node
            next_node += 1

    source = next_node
    sink = next_node + 1
    flow = _Dinic(next_node + 2)

    for key in node_in:
        flow.add_edge(node_in[key], node_out[key], 1)

    for index in range(widths[0]):
        key = (0, index)
        if key in node_in:
            flow.add_edge(source, node_in[key], 1)

    last_layer = len(widths) - 1
    for index in range(widths[-1]):
        key = (last_layer, index)
        if key in node_out:
            flow.add_edge(node_out[key], sink, 1)

    for layer, support in enumerate(support_tuple):
        destinations, sources = np.nonzero(support)
        for destination, source_index in zip(destinations.tolist(), sources.tolist()):
            left = (layer, source_index)
            right = (layer + 1, destination)
            if left in node_out and right in node_in:
                # Unit capacity is enough because every endpoint already has
                # unit vertex capacity.
                flow.add_edge(node_out[left], node_in[right], 1)

    return int(flow.max_flow(source, sink))


def structural_path_certificate(
    supports: Sequence[Array],
    *,
    vertex_masks: Sequence[Array] | None = None,
) -> StructuralPathCertificate:
    """Compute path rank, structural nullity, CDR and simple isolation counts."""

    support_tuple, masks, widths = _validate_supports(supports, vertex_masks)
    path_rank = structural_path_rank(support_tuple, vertex_masks=masks)
    input_dim = widths[0]
    output_dim = widths[-1]
    path_nullity = max(0, input_dim - path_rank)
    cdr = float(path_nullity / input_dim) if input_dim else 0.0

    first = support_tuple[0].copy()
    first &= masks[1][:, None]
    first &= masks[0][None, :]
    active_input_degrees = np.count_nonzero(first, axis=0)
    first_isolated = int(
        np.count_nonzero(masks[0] & (active_input_degrees == 0))
    )

    last = support_tuple[-1].copy()
    last &= masks[-1][:, None]
    last &= masks[-2][None, :]
    active_output_degrees = np.count_nonzero(last, axis=1)
    last_isolated = int(
        np.count_nonzero(masks[-1] & (active_output_degrees == 0))
    )

    return StructuralPathCertificate(
        layer_widths=widths,
        path_rank=path_rank,
        input_dim=input_dim,
        output_dim=output_dim,
        path_nullity=path_nullity,
        collision_deficiency_ratio=cdr,
        active_vertices_per_layer=tuple(int(np.count_nonzero(mask)) for mask in masks),
        first_layer_isolated_inputs=first_isolated,
        last_layer_isolated_outputs=last_isolated,
    )
