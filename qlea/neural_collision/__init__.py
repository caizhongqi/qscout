"""Neural collision certificates for piecewise-affine networks.

The package separates three levels of evidence:

1. numerical fixed-region rank/nullspace certificates;
2. structural path-rank certificates for sparse independently weighted layers;
3. critical-random-regime quantum isolation-query estimates.

Only the first two certify exact structural/local collisions.  The critical
quantum detector targets the asymptotically dominant isolation obstruction near
the random bipartite matching threshold and is not a worst-case replacement for
perfect matching.
"""

from .core import (
    CollisionWitness,
    RegionCertificate,
    analyze_relu_mlp,
    construct_collision,
    forward_relu_mlp,
)
from .quantum import (
    CriticalIsolationResources,
    GroverEstimate,
    asymptotic_perfect_matching_probability,
    classical_isolated_vertices,
    critical_edge_probability,
    critical_isolation_resource_estimate,
    grover_success_probability,
    has_isolation_obstruction,
    optimal_grover_iterations,
)
from .random_graph import (
    PhaseTransitionPoint,
    estimate_phase_transition_point,
    maximum_bipartite_matching_size,
    phase_transition_sweep,
    sample_layered_supports,
)
from .structure import (
    StructuralPathCertificate,
    structural_path_certificate,
    structural_path_rank,
)

__all__ = [
    "CollisionWitness",
    "CriticalIsolationResources",
    "GroverEstimate",
    "PhaseTransitionPoint",
    "RegionCertificate",
    "StructuralPathCertificate",
    "analyze_relu_mlp",
    "asymptotic_perfect_matching_probability",
    "classical_isolated_vertices",
    "construct_collision",
    "critical_edge_probability",
    "critical_isolation_resource_estimate",
    "estimate_phase_transition_point",
    "forward_relu_mlp",
    "grover_success_probability",
    "has_isolation_obstruction",
    "maximum_bipartite_matching_size",
    "optimal_grover_iterations",
    "phase_transition_sweep",
    "sample_layered_supports",
    "structural_path_certificate",
    "structural_path_rank",
]
