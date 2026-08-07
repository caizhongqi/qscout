from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from qlea.neural_collision.torch_adapter import (  # noqa: E402
    analyze_torch_module,
    construct_torch_collision,
)


def test_piecewise_linear_torch_module_collision() -> None:
    module = torch.nn.Sequential(
        torch.nn.Linear(3, 3, bias=False, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 3, bias=False, dtype=torch.float64),
    )
    with torch.no_grad():
        module[0].weight.copy_(torch.eye(3, dtype=torch.float64))
        module[2].weight.copy_(torch.diag(torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64)))

    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    certificate = analyze_torch_module(module, x)

    assert certificate.rank == 2
    assert certificate.nullity == 1
    assert certificate.collision_deficiency_ratio == pytest.approx(1.0 / 3.0)

    witness = construct_torch_collision(
        module,
        x,
        certificate,
        target_l2_shift=0.05,
        output_atol=1e-12,
        output_rtol=1e-12,
    )

    assert witness.numerically_verified
    assert witness.same_relu_signature is True
    assert witness.l2_shift == pytest.approx(0.05, rel=1e-10, abs=1e-12)
    assert witness.output_error_linf < 1e-12
    assert not np.allclose(witness.x, witness.x_prime, rtol=0.0, atol=1e-14)
