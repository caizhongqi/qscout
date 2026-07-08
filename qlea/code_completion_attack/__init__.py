"""Black-box code-completion attack benchmark."""

__all__ = ["CodeAttackConfig", "run_code_attack_benchmark"]


def __getattr__(name: str):
    if name in {"CodeAttackConfig", "run_code_attack_benchmark"}:
        from .benchmark import CodeAttackConfig, run_code_attack_benchmark

        return {
            "CodeAttackConfig": CodeAttackConfig,
            "run_code_attack_benchmark": run_code_attack_benchmark,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
