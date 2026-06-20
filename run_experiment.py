from __future__ import annotations

from qlea import QuantumLoRAExtractor, make_synthetic_task


def run_one(noise_kind: str, noise_p: float) -> dict[str, float]:
    target, x_train, x_test, _y_train, y_test = make_synthetic_task(n_samples=360)
    extractor = QuantumLoRAExtractor(
        n_qubits=3,
        n_layers=2,
        rank=2,
        noise_kind=noise_kind,
        noise_p=noise_p,
    )
    result = extractor.fit(target, x_train[:96], epochs=6)
    metrics = extractor.evaluate(target, result, x_test, y_test)
    metrics["noise_p"] = noise_p
    return metrics


def main() -> None:
    settings = [
        ("none", 0.0),
        ("phase_flip", 0.01),
        ("amplitude_damping", 0.01),
    ]
    print("LoRA-QEA hard-label extraction experiment")
    print("=" * 48)
    for kind, p in settings:
        metrics = run_one(kind, p)
        label = f"{kind}@{p:.3f}"
        print(f"\n[{label}]")
        for key, value in metrics.items():
            if key != "noise_p":
                print(f"{key:28s}: {value:.4f}")


if __name__ == "__main__":
    main()
