# QScout: Query-Guided Noisy Quantum Surrogate for Hard-Label Extraction

QScout is a reproducible research prototype for evaluating whether a physically valid variational quantum circuit (VQC) can guide query-efficient hard-label extraction against classical neural networks.

## Scope

The repository studies three falsifiable questions:

1. Can a compact noisy QNN serve as a hard-label surrogate?
2. Can QNN-guided querying improve a downstream classical clone under a matched query budget?
3. How do capacity, finite shots, noise, data modality, victim architecture, and classical-active baselines affect the result?

It does **not** claim that measurement collapse reveals private training data, or that current pilot results demonstrate a general quantum advantage.

## Quick Start

```powershell
& "D:\\ProgramData\\py2\\python.exe" -m pip install -r requirements.txt
& "D:\\ProgramData\\py2\\python.exe" run_active_hardlabel_benchmark.py --datasets MNIST --victims cnn --strategies random active classical_active --budgets 64 --seeds 7 --qubits 3 --layers 1 --features 8 --victim-epochs 1 --final-epochs 1 --eval-samples 100 --output qscout_smoke.csv
```

## Quantum Contribution

- Physically valid density-matrix VQC simulator.
- Trainable RX/RY/RZ variational circuit with data re-uploading.
- Feature cycling across layers, configurable entanglement, and Z/ZZ readout.
- Phase-flip, bit-flip, and amplitude-damping noise channels.
- Query-guided hard-label extraction with a matched classical-active baseline.
- IBM Quantum replay runner that requires a real user token and never fabricates hardware measurements.

## Evaluation

The full protocol covers MNIST, FashionMNIST, FordA, Wafer, and ElectricDevices; CNN/MLP image victims; 1D-CNN/LSTM time-series victims; query budgets from 64 to 1024; five random seeds; and random, QNN-guided, and classical-active policies.

See `REPRODUCIBILITY.md` and `QSCOUT_EXPERIMENT_PROTOCOL.md` after the repository upload completes.

## License

MIT.