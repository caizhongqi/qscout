"""Quantum low-rank extraction attack prototype."""

from .attack import GeneralQuantumExtractor, QuantumLoRAExtractor
from .target import LoRATarget, make_synthetic_task

__all__ = [
    "GeneralQuantumExtractor",
    "LoRATarget",
    "QuantumLoRAExtractor",
    "make_synthetic_task",
]
