"""Neural collision certificates for piecewise-affine networks.

The core result implemented here is deliberately local and exact: inside the
interior of a fixed ReLU activation region, an MLP is affine.  If the affine
map has a non-trivial kernel, moving along a kernel direction without crossing
an activation boundary produces two distinct inputs with exactly the same
network representation (up to floating-point error).
"""

from .core import (
    CollisionWitness,
    RegionCertificate,
    analyze_relu_mlp,
    construct_collision,
    forward_relu_mlp,
)
from .structure import (
    StructuralPathCertificate,
    structural_path_certificate,
    structural_path_rank,
)

__all__ = [
    "CollisionWitness",
    "RegionCertificate",
    "StructuralPathCertificate",
    "analyze_relu_mlp",
    "construct_collision",
    "forward_relu_mlp",
    "structural_path_certificate",
    "structural_path_rank",
]
