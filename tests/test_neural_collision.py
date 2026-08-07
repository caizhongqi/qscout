from __future__ import annotations

import numpy as np
import pytest

from qlea.neural_collision import (
    analyze_relu_mlp,
    construct_collision,
    structural_path_certificate,
    structural_path_rank,
)


def test_exact_collision_from_rank_deficient_region() -> None:
    weights = [
        np.eye(3),
        np.diag([2.0, -1.0, 0.0]),
    ]
    biases = [np.zeros(3), np.array([0.1, 0.2, -0.3])]
    x = np.array([1.0, 2.0, 3.0])

    certificate = analyze_relu_mlp(weights, biases, x)

    assert certificate.rank == 2
    assert certificate.nullity == 1
    assert certificate.collision_deficiency_ratio == pytest.approx(1.0 / 3.0)
    assert certificate.has_collision_direction

    witness = construct_collision(
        weights,
        biases,
        x,
        certificate,
        target_l2_shift=0.05,
    )

    assert witness.l2_shift == pytest.approx(0.05, rel=1e-10, abs=1e-12)
    assert witness.same_activation_region
    assert witness.kernel_residual_l2 < 1e-12
    assert witness.output_error_linf < 1e-12
    assert not np.allclose(witness.x, witness.x_prime, rtol=0.0, atol=1e-14)


def test_full_rank_region_has_no_local_collision_direction() -> None:
    weights = [np.eye(3), np.eye(3)]
    biases = [np.zeros(3), np.zeros(3)]
    x = np.array([1.0, 2.0, 3.0])

    certificate = analyze_relu_mlp(weights, biases, x)

    assert certificate.rank == 3
    assert certificate.nullity == 0
    assert certificate.locally_injective
    with pytest.raises(ValueError, match="trivial kernel"):
        construct_collision(weights, biases, x, certificate)


def test_inactive_relu_creates_a_collision_direction() -> None:
    weights = [np.eye(3), np.eye(3)]
    biases = [np.zeros(3), np.zeros(3)]
    x = np.array([1.0, 2.0, -1.0])

    certificate = analyze_relu_mlp(weights, biases, x)
    witness = construct_collision(weights, biases, x, certificate, target_l2_shift=0.05)

    assert certificate.rank == 2
    assert certificate.nullity == 1
    assert np.array_equal(certificate.activation_masks[0], [True, True, False])
    assert witness.same_activation_region
    assert witness.output_error_linf < 1e-12


def test_boundary_point_is_not_claimed_as_open_region_certificate() -> None:
    weights = [np.eye(2), np.eye(2)]
    biases = [np.zeros(2), np.zeros(2)]
    x = np.array([1.0, 0.0])

    certificate = analyze_relu_mlp(weights, biases, x)

    assert certificate.on_activation_boundary
    assert not certificate.has_collision_direction
    with pytest.raises(ValueError, match="ReLU boundary"):
        construct_collision(weights, biases, x, certificate)


def test_structural_path_rank_detects_hall_deficit() -> None:
    # Inputs 0 and 1 both have to pass through the same middle vertex, so only
    # two of the three inputs can be routed vertex-disjointly to the output.
    first = np.array(
        [
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    )
    second = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ],
        dtype=bool,
    )

    certificate = structural_path_certificate([first, second])

    assert certificate.path_rank == 2
    assert certificate.path_nullity == 1
    assert certificate.collision_deficiency_ratio == pytest.approx(1.0 / 3.0)
    assert certificate.structurally_collision_prone


def test_structural_path_rank_respects_activation_masks() -> None:
    first = np.eye(3, dtype=bool)
    second = np.eye(3, dtype=bool)
    masks = [
        np.ones(3, dtype=bool),
        np.array([True, True, False]),
        np.ones(3, dtype=bool),
    ]

    assert structural_path_rank([first, second], vertex_masks=masks) == 2


def test_generic_numeric_rank_matches_path_rank_on_sparse_layered_map() -> None:
    rng = np.random.default_rng(13)
    first_support = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    second_support = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    path_rank = structural_path_rank([first_support, second_support])

    w1 = rng.normal(size=first_support.shape) * first_support
    w2 = rng.normal(size=second_support.shape) * second_support
    numeric = analyze_relu_mlp(
        [w1, w2],
        [np.zeros(4), np.zeros(3)],
        np.array([0.2, -0.7, 1.1]),
        relu_after=[False, False],
    )

    assert path_rank == 3
    assert numeric.rank == path_rank
