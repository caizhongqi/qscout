"""Build and execute the QScout main study, ablations, and noise sweeps."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "main", "ablations", "noise", "all"], default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--victim-epochs", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=14)
    args = parser.parse_args()
    common = ["--strategies", "random", "active", "classical_active", "--victim-epochs", str(args.victim_epochs), "--final-epochs", str(args.final_epochs), "--min-victim-accuracy", "0.80"]
    plans: list[tuple[str, list[str]]] = []
    if args.mode in {"smoke", "all"}:
        plans.append(("smoke", ["--datasets", "MNIST", "--victims", "cnn", "--budgets", "64", "--seeds", "7", "--qubits", "3", "--layers", "1", "--features", "8", "--eval-samples", "200"]))
    if args.mode in {"main", "all"}:
        plans.append(("main", ["--datasets", "MNIST", "FashionMNIST", "FordA", "Wafer", "ElectricDevices", "--victims", "cnn", "mlp", "cnn1d", "lstm", "--budgets", "64,128,256,512,1024", "--seeds", "7,19,31,43,59", "--qubits", "4", "--layers", "3", "--features", "16"]))
    if args.mode in {"ablations", "all"}:
        plans.append(("capacity", ["--datasets", "MNIST", "Wafer", "--victims", "cnn", "cnn1d", "--budgets", "256", "--seeds", "7,19,31,43,59", "--qubits-list", "2,3,4,5", "--layers-list", "1,2,3", "--features-list", "8,12,16,20"]))
    if args.mode in {"noise", "all"}:
        for noise in ("phase_flip", "bit_flip", "amplitude_damping"):
            for probability in ("0.002", "0.005", "0.01"):
                plans.append((f"noise_{noise}_{probability}", ["--datasets", "MNIST", "Wafer", "--victims", "cnn", "cnn1d", "--budgets", "256", "--seeds", "7,19,31,43,59", "--qubits", "4", "--layers", "3", "--features", "16", "--noise-kind", noise, "--noise-p", probability]))
    for name, extra in plans:
        command = [sys.executable, "run_active_hardlabel_benchmark.py", *common, *extra, "--output", f"study_{name}.csv"]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
