"""PyTorch adapter for local collision analysis on trained neural networks.

This adapter is intentionally separated from the NumPy core because PyTorch is
an optional dependency of QScout.  It supports arbitrary differentiable
``torch.nn.Module`` objects and is therefore the bridge to trained MLP/CNN
representations.

Scope of the certificate
------------------------
For a ReLU/MaxPool/affine network in evaluation mode, the map is piecewise
affine.  A rank-deficient Jacobian at an interior point gives the same local
kernel mechanism as the exact NumPy certificate.  For smooth nonlinear models
(GELU, sigmoid, attention softmax, etc.), a Jacobian nullspace is only a
first-order statement; this module therefore calls its constructed pair a
*numerically verified* collision only when direct forward evaluation confirms
the requested tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class TorchJacobianCertificate:
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dim: int
    output_dim: int
    rank: int
    nullity: int
    collision_deficiency_ratio: float
    singular_values: Array
    rank_tolerance: float
    jacobian: Array
    nullspace_basis: Array

    @property
    def has_null_direction(self) -> bool:
        return self.nullity > 0


@dataclass(frozen=True)
class TorchCollisionWitness:
    x: Array
    x_prime: Array
    direction: Array
    l2_shift: float
    output_error_l2: float
    output_error_linf: float
    accepted_tolerance: float
    numerically_verified: bool
    same_relu_signature: bool | None
    backtracking_steps: int


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "PyTorch is required for qlea.neural_collision.torch_adapter; "
            "install qscout[llm] or qscout[all]"
        ) from exc
    return torch


def _matrix_rank_nullspace(
    matrix: Array,
    rank_tolerance: float | None,
) -> tuple[int, Array, Array, float]:
    rows, cols = matrix.shape
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    if rank_tolerance is None:
        largest = float(singular_values[0]) if singular_values.size else 0.0
        rank_tolerance = max(rows, cols) * np.finfo(float).eps * max(1.0, largest)
    if rank_tolerance < 0:
        raise ValueError("rank_tolerance must be non-negative")
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    nullspace = vh[rank:, :].T.copy()
    return rank, singular_values.copy(), nullspace, float(rank_tolerance)


def analyze_torch_module(
    module: Any,
    x: Any,
    *,
    rank_tolerance: float | None = None,
    vectorize: bool = True,
) -> TorchJacobianCertificate:
    """Compute a flattened output-vs-input Jacobian rank certificate.

    ``x`` should represent a single example.  A leading batch dimension of one
    is allowed and simply becomes part of the flattened input coordinates.
    The analyzed map is exactly ``module(x)``; to study an internal
    representation, pass the corresponding feature submodule or a wrapper
    module whose ``forward`` returns that representation.
    """

    torch = _torch()
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not x.is_floating_point():
        raise TypeError("x must have a floating-point dtype")

    was_training = module.training
    module.eval()
    try:
        x_work = x.detach().clone().requires_grad_(True)

        def flattened_output(value):
            output = module(value)
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    "module output must be a single torch.Tensor for Jacobian analysis"
                )
            return output.reshape(-1)

        with torch.enable_grad():
            output = module(x_work)
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    "module output must be a single torch.Tensor for Jacobian analysis"
                )
            jacobian = torch.autograd.functional.jacobian(
                flattened_output,
                x_work,
                vectorize=vectorize,
                create_graph=False,
                strict=False,
            )
        matrix = (
            jacobian.detach()
            .reshape(output.numel(), x_work.numel())
            .to(dtype=torch.float64, device="cpu")
            .numpy()
        )
    finally:
        module.train(was_training)

    rank, singular_values, nullspace, effective_tol = _matrix_rank_nullspace(
        matrix, rank_tolerance
    )
    input_dim = int(x.numel())
    output_dim = int(output.numel())
    nullity = input_dim - rank
    return TorchJacobianCertificate(
        input_shape=tuple(int(size) for size in x.shape),
        output_shape=tuple(int(size) for size in output.shape),
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank,
        nullity=nullity,
        collision_deficiency_ratio=float(nullity / input_dim) if input_dim else 0.0,
        singular_values=singular_values,
        rank_tolerance=effective_tol,
        jacobian=matrix,
        nullspace_basis=nullspace,
    )


def relu_module_signature(module: Any, x: Any) -> tuple[Array, ...]:
    """Capture Boolean output signs of explicit ``nn.ReLU`` modules.

    Functional calls such as ``torch.nn.functional.relu`` are not visible to
    module hooks, so an empty signature does not imply that no ReLU operation
    occurred.
    """

    torch = _torch()
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    signatures: list[Array] = []
    handles = []

    def hook(_submodule, _inputs, output):
        if isinstance(output, torch.Tensor):
            signatures.append((output.detach() > 0).to(device="cpu").numpy())

    for submodule in module.modules():
        if isinstance(submodule, torch.nn.ReLU):
            handles.append(submodule.register_forward_hook(hook))

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            module(x)
    finally:
        for handle in handles:
            handle.remove()
        module.train(was_training)
    return tuple(signatures)


def construct_torch_collision(
    module: Any,
    x: Any,
    certificate: TorchJacobianCertificate | None = None,
    *,
    basis_index: int = 0,
    target_l2_shift: float = 5e-2,
    output_atol: float = 1e-7,
    output_rtol: float = 1e-7,
    max_backtracking_steps: int = 30,
    capture_relu_signature: bool = True,
) -> TorchCollisionWitness:
    """Backtrack along a Jacobian-null direction and verify the pair directly.

    For piecewise-affine modules this normally finds a finite step inside the
    same local region.  For smooth nonlinear modules the accepted step can be
    arbitrarily small and should not be interpreted as an exact symbolic
    collision theorem.
    """

    torch = _torch()
    if target_l2_shift <= 0:
        raise ValueError("target_l2_shift must be positive")
    if output_atol < 0 or output_rtol < 0:
        raise ValueError("output tolerances must be non-negative")
    if max_backtracking_steps < 0:
        raise ValueError("max_backtracking_steps must be non-negative")

    if certificate is None:
        certificate = analyze_torch_module(module, x)
    if certificate.nullity <= 0:
        raise ValueError("Jacobian has trivial kernel; no local null direction")
    if tuple(int(size) for size in x.shape) != certificate.input_shape:
        raise ValueError("certificate input shape does not match x")
    if not 0 <= basis_index < certificate.nullspace_basis.shape[1]:
        raise IndexError("basis_index is outside the nullspace basis")

    direction_np = certificate.nullspace_basis[:, basis_index].copy()
    direction_norm = float(np.linalg.norm(direction_np))
    if direction_norm == 0.0:
        raise RuntimeError("nullspace basis contains a zero vector")
    direction_np /= direction_norm
    direction = torch.as_tensor(
        direction_np.reshape(certificate.input_shape),
        dtype=x.dtype,
        device=x.device,
    )

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            y0 = module(x).detach()
        if not isinstance(y0, torch.Tensor):
            raise TypeError("module output must be a torch.Tensor")
        reference_scale = float(torch.linalg.vector_norm(y0.reshape(-1)).item())
        accepted_tolerance = output_atol + output_rtol * reference_scale

        original_signature = (
            relu_module_signature(module, x) if capture_relu_signature else ()
        )

        best = None
        for backtracking in range(max_backtracking_steps + 1):
            magnitude = target_l2_shift * (0.5**backtracking)
            for sign in (1.0, -1.0):
                candidate = x + sign * magnitude * direction
                with torch.no_grad():
                    y1 = module(candidate).detach()
                delta = (y1 - y0).reshape(-1)
                error_l2 = float(torch.linalg.vector_norm(delta).item())
                error_linf = float(torch.max(torch.abs(delta)).item()) if delta.numel() else 0.0
                if best is None or error_l2 < best[0]:
                    best = (error_l2, error_linf, candidate.detach(), backtracking)
                if error_l2 <= accepted_tolerance:
                    candidate_signature = (
                        relu_module_signature(module, candidate)
                        if capture_relu_signature
                        else ()
                    )
                    same_signature: bool | None
                    if not capture_relu_signature or not original_signature:
                        same_signature = None
                    else:
                        same_signature = len(original_signature) == len(candidate_signature) and all(
                            np.array_equal(left, right)
                            for left, right in zip(original_signature, candidate_signature)
                        )
                    x_np = x.detach().to(device="cpu").numpy()
                    candidate_np = candidate.detach().to(device="cpu").numpy()
                    return TorchCollisionWitness(
                        x=x_np,
                        x_prime=candidate_np,
                        direction=direction_np.reshape(certificate.input_shape),
                        l2_shift=float(np.linalg.norm(candidate_np - x_np)),
                        output_error_l2=error_l2,
                        output_error_linf=error_linf,
                        accepted_tolerance=float(accepted_tolerance),
                        numerically_verified=True,
                        same_relu_signature=same_signature,
                        backtracking_steps=backtracking,
                    )
    finally:
        module.train(was_training)

    assert best is not None
    error_l2, error_linf, candidate, backtracking = best
    x_np = x.detach().to(device="cpu").numpy()
    candidate_np = candidate.to(device="cpu").numpy()
    return TorchCollisionWitness(
        x=x_np,
        x_prime=candidate_np,
        direction=direction_np.reshape(certificate.input_shape),
        l2_shift=float(np.linalg.norm(candidate_np - x_np)),
        output_error_l2=float(error_l2),
        output_error_linf=float(error_linf),
        accepted_tolerance=float(accepted_tolerance),
        numerically_verified=False,
        same_relu_signature=None,
        backtracking_steps=int(backtracking),
    )
