"""Baseline registry for the LLM security probing protocol.

The executable implementation lives in `qlea.code_completion_attack.benchmark`
so that all methods share the same candidate pool, detector, and accounting
code.  This registry gives reviewers a stable mapping from paper names to
strategy identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    paper_name: str
    strategy_id: str
    family: str
    quantum: bool
    description: str


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        paper_name="Random Search",
        strategy_id="fair_random_comment",
        family="random",
        quantum=False,
        description="Uniform sampling from the same benchmark-authorized candidate pool.",
    ),
    BaselineSpec(
        paper_name="Risk Prior",
        strategy_id="fair_risk_prior_comment",
        family="lexical prior",
        quantum=False,
        description="Ranks prompts by security-risk and legacy-compatibility lexical priors.",
    ),
    BaselineSpec(
        paper_name="INSEC-style Fixed-Pool Search",
        strategy_id="insec_fixed_pool_comment",
        family="fixed-pool attack prior",
        quantum=False,
        description="Ranks the fair pool by compatibility pressure and risk-prior signals.",
    ),
    BaselineSpec(
        paper_name="Classical Active",
        strategy_id="classical_active_comment",
        family="active learning",
        quantum=False,
        description="Classical embedding uncertainty/diversity active-query baseline.",
    ),
    BaselineSpec(
        paper_name="AOT-style Ensemble",
        strategy_id="aot_ensemble_fixed_pool_comment",
        family="ensemble active search",
        quantum=False,
        description="Fixed-pool ensemble-inspired ranking baseline.",
    ),
    BaselineSpec(
        paper_name="Classical Boundary Witness",
        strategy_id="classical_boundary_witness_comment",
        family="classical boundary witness",
        quantum=False,
        description="Classical boundary witness control without quantum state discrimination.",
    ),
    BaselineSpec(
        paper_name="QScout-QBW",
        strategy_id="qscout_qbw_comment",
        family="objective-aligned quantum query learning",
        quantum=True,
        description="Guarded quantum boundary witness with objective-aligned hard-label feedback.",
    ),
)


def strategy_ids(include_quantum: bool = True) -> tuple[str, ...]:
    return tuple(spec.strategy_id for spec in BASELINES if include_quantum or not spec.quantum)


def csv_strategy_list(include_quantum: bool = True) -> str:
    return ",".join(strategy_ids(include_quantum=include_quantum))
