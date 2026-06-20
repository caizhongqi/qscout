from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from qlea.attack import GeneralQuantumExtractor, QuantumLoRAExtractor
from qlea.datasets import all_datasets
from qlea.models import LoRAVictim, fit_lora_victim, fit_mlp_victim


OUT_DIR = Path("outputs")


def save_visual_samples(bundle) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    if bundle.raw_train is None:
        return
    fig, axes = plt.subplots(2, 5, figsize=(7, 3))
    axes = axes.ravel()
    for ax, raw, label in zip(axes, bundle.raw_train[:10], bundle.y_train[:10]):
        if bundle.name == "digits_8x8_images":
            ax.imshow(raw, cmap="gray")
            ax.set_title(f"y={label}")
            ax.axis("off")
        else:
            ax.plot(raw, linewidth=1.5)
            ax.set_title(f"y={label}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{bundle.name}_samples.png", dpi=160)
    plt.close(fig)


def run_general_attack(victim, bundle, n_classes: int, seed: int = 13) -> dict[str, float]:
    extractor = GeneralQuantumExtractor(n_qubits=3, n_layers=2, seed=seed)
    result = extractor.fit(victim.predict, bundle.x_train[:160], epochs=12, n_classes=n_classes)
    return extractor.evaluate(victim.predict, result, bundle.x_test, bundle.y_test)


def run_lora_attack(victim: LoRAVictim, bundle) -> dict[str, float]:
    extractor = QuantumLoRAExtractor(n_qubits=3, n_layers=2, rank=2, seed=17)
    result = extractor.fit(victim.target, bundle.x_train[:160], epochs=12)
    return extractor.evaluate(victim.target, result, bundle.x_test, bundle.y_test)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    for bundle in all_datasets():
        save_visual_samples(bundle)
        n_classes = int(np.max(bundle.y_train)) + 1
        victims = []
        if bundle.name == "tabular_low_rank":
            victims.append(fit_lora_victim(bundle.x_train, bundle.y_train))
        victims.extend(
            [
                fit_mlp_victim(
                    bundle.x_train,
                    bundle.y_train,
                    name=f"shallow_mlp_{bundle.name}",
                    hidden=(32,),
                ),
                fit_mlp_victim(
                    bundle.x_train,
                    bundle.y_train,
                    name=f"deep_mlp_{bundle.name}",
                    hidden=(48, 24) if n_classes <= 3 else (96, 48),
                ),
            ]
        )

        for victim in victims:
            if isinstance(victim, LoRAVictim):
                metrics = run_lora_attack(victim, bundle)
                attack = "qnn_lora_projection"
            else:
                metrics = run_general_attack(victim, bundle, n_classes)
                attack = "qnn_hard_label_surrogate"
            row = {
                "dataset": bundle.name,
                "victim": victim.name,
                "attack": attack,
                **metrics,
            }
            rows.append(row)

    header = [
        "dataset",
        "victim",
        "attack",
        "victim_accuracy",
        "qnn_extraction_accuracy",
        "recovered_model_agreement",
        "delta_cosine",
        "delta_relative_error",
    ]
    csv_path = OUT_DIR / "benchmark_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(col, "")) for col in header) + "\n")

    print("Benchmark finished")
    print(f"results: {csv_path}")
    for row in rows:
        print(
            f"{row['dataset']:24s} | {row['victim']:24s} | "
            f"victim_acc={row['victim_accuracy']:.3f} | "
            f"qnn_agreement={row['qnn_extraction_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
