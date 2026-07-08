"""QScout package entry point.

The package keeps its top-level import lightweight so utility modules such as
``qlea.theory`` and ``qlea.quantum_boundary_witness`` can be used without
eagerly importing optional experiment stacks such as scikit-learn, torch, or
transformers.
"""

__all__ = [
    "GeneralQuantumExtractor",
    "LoRATarget",
    "QuantumLoRAExtractor",
    "make_synthetic_task",
]


def __getattr__(name: str):
    if name in {"GeneralQuantumExtractor", "QuantumLoRAExtractor"}:
        from .attack import GeneralQuantumExtractor, QuantumLoRAExtractor

        return {
            "GeneralQuantumExtractor": GeneralQuantumExtractor,
            "QuantumLoRAExtractor": QuantumLoRAExtractor,
        }[name]
    if name in {"LoRATarget", "make_synthetic_task"}:
        from .target import LoRATarget, make_synthetic_task

        return {
            "LoRATarget": LoRATarget,
            "make_synthetic_task": make_synthetic_task,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
