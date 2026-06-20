# Reproducibility Guide

## Scope

QScout is a capability-boundary research prototype for hard-label extraction
against classical victims. It does not claim that a quantum circuit can recover
private data from labels, nor that it has established a general quantum
advantage over strong classical attacks.

The repository separates three questions:

1. Can a physically valid, noisy VQC be trained as a hard-label surrogate?
2. Does a QNN-guided query policy improve a downstream classical clone over a
   random-query clone under the same query budget?
3. How do the conclusions change under circuit capacity, finite shots, noise,
   victim architecture, dataset, and defenses?

## Environment

The Windows setup used for the current project is:

```powershell
& "D:\ProgramData\py2\python.exe" -m pip install -r requirements.txt
```

`qiskit` and `qiskit-ibm-runtime` are only needed for an IBM hardware replay.
The main simulator experiments use PyTorch, NumPy, scikit-learn, and the local
density-matrix implementation.

## Quick smoke test

```powershell
& "D:\ProgramData\py2\python.exe" run_active_hardlabel_benchmark.py `
  --datasets MNIST --victims cnn --strategies random active classical_active `
  --budgets 64 --seeds 7 --qubits 3 --layers 1 --features 8 `
  --victim-epochs 1 --final-epochs 1 --eval-samples 100 `
  --output qscout_smoke.csv
```

This command validates the end-to-end path only; its small query budget and
single seed are not publication-quality evidence.

## Full study protocol

Use the pre-registered design in `QSCOUT_EXPERIMENT_PROTOCOL.md`:

- Five public datasets: MNIST, FashionMNIST, FordA, Wafer, and ElectricDevices.
- Victims: CNN/MLP for images and 1D-CNN/LSTM for time series.
- Budgets: 64, 128, 256, 512, and 1024 hard-label queries.
- Seeds: at least five independent runs.
- Query policies: random, classical uncertainty/diversity, and QNN-guided.
- QNN ablations: qubits, layers, feature cycling, entanglement, ZZ readout,
  shots, and noise.

Do not report a QNN-guided improvement as a quantum advantage unless its
confidence interval exceeds both the majority-class and matched classical-active
baselines on the same victims, splits, budgets, and random seeds.

## IBM QPU replay

Set an IBM Quantum token only in the shell session, never in a tracked file:

```powershell
$env:QISKIT_IBM_TOKEN = "your-token"
& "D:\ProgramData\py2\python.exe" run_ibm_qpu_experiment.py
```

See `REAL_QPU_EXPERIMENT.md` for caveats. No hardware claim is valid until the
job ID, backend calibration, shots, transpilation settings, and raw result CSV
are archived.
