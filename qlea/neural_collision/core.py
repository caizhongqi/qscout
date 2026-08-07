"""Exact local collision certificates for feed-forward ReLU networks.

For a fixed activation pattern, a ReLU MLP is an affine map

    f(x) = A_R x + b_R.

If ``rank(A_R) < input_dim`` and ``x`` lies in the interior of the activation
region, every non-zero vector ``v`` in ``ker(A_R)`` is a collision direction.
A sufficiently small non-zero step ``t`` that does not cross any ReLU boundary
therefore satisfies

    f(x + t v) = f(x),   x + t v != x.

This module computes the affine map, a numerical rank/nullspace certificate,
and a boundary-safe collision witness.  The implementation is NumPy-only so it
can be used in the lightweight artifact environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class RegionCertificate:
    """Numerical certificate for one ReLU activation region.

    Attributes
    ----------
    affine_matrix:
        Local affine matrix ``A_R`` with shape ``(output_dim, input_dim)``.
    affine_bias:
        Local affine offset ``b_R``.
    rank:
        Numerical rank of ``A_R`` under ``rank_tolerance``.
    nullity:
        ``input_dim - rank``.
    collision_deficiency_ratio:
        ``nullity / input_dim``.  This is the CDR used by the experiments.
    nullspace_basis:
        Orthonormal basis of ``ker(A_R)`` stored column-wise.
    preactivations / activation_masks:
        ReLU preactivations and their fixed binary masks, in layer order.
    min_abs_preactivation:
        Distance to the nearest ReLU boundary in preactivation coordinates.
        A value below ``boundary_tolerance`` means the point is numerically on
        an activation boundary and no open fixed-region certificate is claimed.
    """

    input_dim: int
    output_dim: int
    affine_matrix: Array
    affine_bias: Array
    output: Array
    rank: int
    nullity: int
    collision_deficiency_ratio: float
    singular_values: Array
    rank_tolerance: float
    nullspace_basis: Array
    preactivations: tuple[Array, ...]
    activation_masks: tuple[Array, ...]
    relu_after: tuple[bool, ...]
    min_abs_preactivation: float
    boundary_tolerance: float
    on_activation_boundary: bool

    @property
    def locally_injective(self) -> bool:
        """Whether the fixed-region affine map is injective on input space."""

        return self.rank == self.input_dim

    @property
    def has_collision_direction(self) -> bool:
        """Whether a non-trivial local kernel direction is certified."""

        return self.nullity > 0 and not self.on_activation_boundary


@dataclass(frozen=True)
class CollisionWitness:
    """A concrete pair of distinct inputs with the same local representation."""

    x: Array
    x_prime: Array
    direction: Array
    step: float
    l2_shift: float
    kernel_residual_l2: float
    output_error_l2: float
    output_error_linf: float
    same_activation_region: bool


def _as_float_array(value: Array, *, ndim: int, name: str) -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {arr.shape}")
    return arr


def _validate_network(
    weights: Sequence[Array],
    biases: Sequence[Array],
    x: Array,
    relu_after: Sequence[bool] | None,
) -> tuple[tuple[Array, ...], tuple[Array, ...], Array, tuple[bool, ...]]:
    if not weights:
        raise ValueError("weights must contain at least one affine layer")
    if len(weights) != len(biases):
        raise ValueError("weights and biases must have the same length")

    w_tuple = tuple(
        _as_float_array(weight, ndim=2, name=f"weights[{index}]")
        for index, weight in enumerate(weights)
    )
    b_tuple = tuple(
        _as_float_array(bias, ndim=1, name=f"biases[{index}]")
        for index, bias in enumerate(biases)
    )
    x_arr = _as_float_array(x, ndim=1, name="x")

    previous_dim = x_arr.shape[0]
    for index, (weight, bias) in enumerate(zip(w_tuple, b_tuple)):
        if weight.shape[1] != previous_dim:
            raise ValueError(
                f"weights[{index}] expects {weight.shape[1]} inputs, "
                f"but previous layer has width {previous_dim}"
            )
        if weight.shape[0] != bias.shape[0]:
            raise ValueError(
                f"biases[{index}] has length {bias.shape[0]}, expected {weight.shape[0]}"
            )
        previous_dim = weight.shape[0]

    if relu_after is None:
        relu_tuple = tuple(index < len(w_tuple) - 1 for index in range(len(w_tuple)))
    else:
        relu_tuple = tuple(bool(flag) for flag in relu_after)
        if len(relu_tuple) != len(w_tuple):
            raise ValueError("relu_after must have one boolean per affine layer")

    return w_tuple, b_tuple, x_arr, relu_tuple


def forward_relu_mlp(
    weights: Sequence[Array],
    biases: Sequence[Array],
    x: Array,
    *,
    relu_after: Sequence[bool] | None = None,
) -> Array:
    """Evaluate a feed-forward affine/ReLU network."""

    w_tuple, b_tuple, hidden, relu_tuple = _validate_network(
        weights, biases, x, relu_after
    )
    for weight, bias, use_relu in zip(w_tuple, b_tuple, relu_tuple):
        hidden = weight @ hidden + bias
        if use_relu:
            hidden = np.maximum(hidden, 0.0)
    return hidden


def _rank_and_nullspace(matrix: Array, rank_tolerance: float | None) -> tuple[int, Array, Array, float]:
    """Return rank, singular values, nullspace basis and effective tolerance."""

    rows, cols = matrix.shape
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    if rank_tolerance is None:
        largest = float(singular_values[0]) if singular_values.size else 0.0
        rank_tolerance = max(rows, cols) * np.finfo(float).eps * max(largest, 1.0)
    if rank_tolerance < 0:
        raise ValueError("rank_tolerance must be non-negative")
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    nullspace = vh[rank:, :].T.copy()
    return rank, singular_values.copy(), nullspace, float(rank_tolerance)


def analyze_relu_mlp(
    weights: Sequence[Array],
    biases: Sequence[Array],
    x: Array,
    *,
    relu_after: Sequence[bool] | None = None,
    rank_tolerance: float | None = None,
    boundary_tolerance: float = 1e-10,
) -> RegionCertificate:
    """Compute the exact affine representation on the activation region of ``x``.

    The result is a *local* certificate.  When ``nullity > 0`` and ``x`` is not
    on an activation boundary, :func:`construct_collision` can turn the
    nullspace into an explicit collision pair.
    """

    if boundary_tolerance < 0:
        raise ValueError("boundary_tolerance must be non-negative")

    w_tuple, b_tuple, x_arr, relu_tuple = _validate_network(
        weights, biases, x, relu_after
    )

    input_dim = int(x_arr.shape[0])
    affine_matrix = np.eye(input_dim, dtype=float)
    affine_bias = np.zeros(input_dim, dtype=float)
    hidden = x_arr.copy()

    preactivations: list[Array] = []
    masks: list[Array] = []

    for weight, bias, use_relu in zip(w_tuple, b_tuple, relu_tuple):
        preactivation = weight @ hidden + bias
        next_matrix = weight @ affine_matrix
        next_bias = weight @ affine_bias + bias

        if use_relu:
            mask = preactivation > 0.0
            preactivations.append(preactivation.copy())
            masks.append(mask.copy())
            hidden = np.where(mask, preactivation, 0.0)
            affine_matrix = mask[:, None] * next_matrix
            affine_bias = mask * next_bias
        else:
            hidden = preactivation
            affine_matrix = next_matrix
            affine_bias = next_bias

    output = hidden.copy()
    reconstructed = affine_matrix @ x_arr + affine_bias
    if not np.allclose(output, reconstructed, rtol=1e-10, atol=1e-12):
        raise RuntimeError("internal error: fixed-region affine reconstruction failed")

    rank, singular_values, nullspace, effective_tol = _rank_and_nullspace(
        affine_matrix, rank_tolerance
    )
    nullity = input_dim - rank
    cdr = float(nullity / input_dim) if input_dim else 0.0

    if preactivations:
        min_abs = min(float(np.min(np.abs(z))) for z in preactivations)
    else:
        min_abs = float("inf")
    on_boundary = bool(min_abs <= boundary_tolerance)

    return RegionCertificate(
        input_dim=input_dim,
        output_dim=int(output.shape[0]),
        affine_matrix=affine_matrix.copy(),
        affine_bias=affine_bias.copy(),
        output=output,
        rank=rank,
        nullity=nullity,
        collision_deficiency_ratio=cdr,
        singular_values=singular_values,
        rank_tolerance=effective_tol,
        nullspace_basis=nullspace,
        preactivations=tuple(preactivations),
        activation_masks=tuple(masks),
        relu_after=relu_tuple,
        min_abs_preactivation=min_abs,
        boundary_tolerance=float(boundary_tolerance),
        on_activation_boundary=on_boundary,
    )


def _directional_preactivations(
    weights: tuple[Array, ...],
    relu_after: tuple[bool, ...],
    masks: tuple[Array, ...],
    direction: Array,
) -> tuple[Array, ...]:
    derivative = direction.copy()
    result: list[Array] = []
    mask_index = 0
    for weight, use_relu in zip(weights, relu_after):
        derivative = weight @ derivative
        if use_relu:
            result.append(derivative.copy())
            derivative = masks[mask_index] * derivative
            mask_index += 1
    return tuple(result)


def _fixed_region_step_interval(
    preactivations: tuple[Array, ...],
    directional_preactivations: tuple[Array, ...],
    *,
    direction_tolerance: float,
) -> tuple[float, float]:
    """Find the open interval of steps that preserves every ReLU sign."""

    lower = -float("inf")
    upper = float("inf")

    for z_values, dz_values in zip(preactivations, directional_preactivations):
        for z, dz in zip(z_values, dz_values):
            z = float(z)
            dz = float(dz)
            if abs(dz) <= direction_tolerance:
                continue
            if z == 0.0:
                return 0.0, 0.0
            crossing = -z / dz
            if z > 0.0:
                if dz > 0.0:
                    lower = max(lower, crossing)
                else:
                    upper = min(upper, crossing)
            else:
                if dz > 0.0:
                    upper = min(upper, crossing)
                else:
                    lower = max(lower, crossing)

    return lower, upper


def _choose_step(
    lower: float,
    upper: float,
    *,
    target_shift: float,
    safety_fraction: float,
    minimum_step: float,
) -> float:
    if not (lower < 0.0 < upper):
        raise RuntimeError(
            "the null direction has no open fixed-activation interval around x"
        )

    def safe_capacity(bound: float) -> float:
        if np.isinf(bound):
            return float("inf")
        return max(0.0, safety_fraction * bound)

    positive_capacity = safe_capacity(upper)
    negative_capacity = safe_capacity(-lower)

    preferred_sign = 1.0 if positive_capacity >= negative_capacity else -1.0
    capacities = (
        (preferred_sign, positive_capacity if preferred_sign > 0 else negative_capacity),
        (-preferred_sign, negative_capacity if preferred_sign > 0 else positive_capacity),
    )
    for sign, capacity in capacities:
        magnitude = target_shift if np.isinf(capacity) else min(target_shift, capacity)
        if magnitude >= minimum_step:
            return sign * magnitude

    raise RuntimeError(
        "activation region is too narrow to construct a numerically distinct collision"
    )


def construct_collision(
    weights: Sequence[Array],
    biases: Sequence[Array],
    x: Array,
    certificate: RegionCertificate | None = None,
    *,
    relu_after: Sequence[bool] | None = None,
    target_l2_shift: float = 5e-2,
    safety_fraction: float = 0.5,
    minimum_step: float = 1e-12,
    direction_tolerance: float = 1e-14,
    basis_index: int = 0,
) -> CollisionWitness:
    """Construct a concrete exact collision inside the current ReLU region.

    ``target_l2_shift`` is an upper target, not a promise: if the closest ReLU
    boundary is nearer, the routine takes a smaller boundary-safe step.
    """

    if target_l2_shift <= 0:
        raise ValueError("target_l2_shift must be positive")
    if not 0.0 < safety_fraction < 1.0:
        raise ValueError("safety_fraction must lie strictly between 0 and 1")
    if minimum_step <= 0:
        raise ValueError("minimum_step must be positive")

    w_tuple, b_tuple, x_arr, relu_tuple = _validate_network(
        weights, biases, x, relu_after
    )
    if certificate is None:
        certificate = analyze_relu_mlp(
            w_tuple, b_tuple, x_arr, relu_after=relu_tuple
        )

    if certificate.input_dim != x_arr.shape[0]:
        raise ValueError("certificate input dimension does not match x")
    if certificate.relu_after != relu_tuple:
        raise ValueError("certificate relu_after pattern does not match network")
    if certificate.on_activation_boundary:
        raise ValueError(
            "x is numerically on a ReLU boundary; no open-region collision is certified"
        )
    if certificate.nullity <= 0:
        raise ValueError("fixed-region map has trivial kernel; no local collision direction")
    if not 0 <= basis_index < certificate.nullspace_basis.shape[1]:
        raise IndexError("basis_index is outside the certified nullspace basis")

    direction = certificate.nullspace_basis[:, basis_index].copy()
    norm = float(np.linalg.norm(direction))
    if norm <= direction_tolerance:
        raise RuntimeError("numerical nullspace basis contains a zero direction")
    direction /= norm

    directional_z = _directional_preactivations(
        w_tuple,
        relu_tuple,
        certificate.activation_masks,
        direction,
    )
    lower, upper = _fixed_region_step_interval(
        certificate.preactivations,
        directional_z,
        direction_tolerance=direction_tolerance,
    )
    step = _choose_step(
        lower,
        upper,
        target_shift=target_l2_shift,
        safety_fraction=safety_fraction,
        minimum_step=minimum_step,
    )

    x_prime = x_arr + step * direction
    output_x = forward_relu_mlp(w_tuple, b_tuple, x_arr, relu_after=relu_tuple)
    output_prime = forward_relu_mlp(w_tuple, b_tuple, x_prime, relu_after=relu_tuple)
    error = output_prime - output_x

    prime_certificate = analyze_relu_mlp(
        w_tuple,
        b_tuple,
        x_prime,
        relu_after=relu_tuple,
        rank_tolerance=certificate.rank_tolerance,
        boundary_tolerance=certificate.boundary_tolerance,
    )
    same_region = all(
        np.array_equal(left, right)
        for left, right in zip(
            certificate.activation_masks, prime_certificate.activation_masks
        )
    ) and len(certificate.activation_masks) == len(prime_certificate.activation_masks)

    return CollisionWitness(
        x=x_arr.copy(),
        x_prime=x_prime,
        direction=direction,
        step=float(step),
        l2_shift=float(np.linalg.norm(x_prime - x_arr)),
        kernel_residual_l2=float(np.linalg.norm(certificate.affine_matrix @ direction)),
        output_error_l2=float(np.linalg.norm(error)),
        output_error_linf=float(np.max(np.abs(error))) if error.size else 0.0,
        same_activation_region=bool(same_region),
    )
