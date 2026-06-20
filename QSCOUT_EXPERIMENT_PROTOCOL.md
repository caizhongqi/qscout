# Q-Scout Top-Conference Experiment Protocol

## Final configuration

The current development winner is **4 qubits, 3 layers, 16 feature-cycling
components, circular entanglement, Z+ZZ readout**. It is a development
candidate, not a final claim, until selected by multi-seed validation. The
five-qubit pilot performed worse and belongs in the capacity ablation.

## Main table

- Datasets: MNIST, FashionMNIST, FordA, Wafer, ElectricDevices.
- Victims: high-quality CNN/MLP or 1D-CNN/LSTM only; skip any victim below the
  configured accuracy threshold.
- Methods: Random clone, Classical-Active clone, Q-Scout-guided clone. Add a
  faithful QEDG/MAZE reproduction only after matching its published threat
  model and query budget.
- Budgets: 64, 128, 256, 512, 1024, 2048.
- Seeds: 7, 19, 31, 43, 59.
- Metrics: functional fidelity, victim-correct fidelity, majority-class gain,
  clone accuracy, query cost, and 95% confidence intervals.

## Ablations

- Qubits: 2, 3, 4, 5, 6.
- Layers: 1, 2, 3, 4.
- Encoding: feature cycling on/off; 8, 12, 16, 20, 24 components.
- Measurements: Z only versus Z+ZZ.
- Entanglement: none, linear, circular, star, full.
- Noise: ideal, phase flip, amplitude damping, depolarizing/readout models.
- Hardware: analytic simulation, 512/1024/2048/4096 shots, then real QPU.

## Defense stress test

Compare the same attacks under FlowGuard-style content OOD, PRADA, FDINet, and
single-account versus Sybil-distributed queries. Report fidelity-detection
curves, not only a detector ROC or an attack score in isolation.

## Reproducible commands

Development capacity sweep on MNIST:

```powershell
& "D:\ProgramData\py2\python.exe" run_active_hardlabel_benchmark.py `
  --datasets MNIST --victims cnn --strategies random active classical_active `
  --budgets 128,256,512 --seeds 7,19,31 `
  --qubits-list 2,3,4,5 --layers-list 1,2,3 --features-list 8,12,16,20 `
  --victim-epochs 5 --final-epochs 8 --output qscout_mnist_capacity.csv
```

Main study with the validation-selected configuration:

```powershell
& "D:\ProgramData\py2\python.exe" run_active_hardlabel_benchmark.py `
  --datasets MNIST FashionMNIST FordA Wafer ElectricDevices `
  --victims cnn cnn1d --strategies random active classical_active `
  --budgets 64,128,256,512,1024,2048 --seeds 7,19,31,43,59 `
  --qubits 4 --layers 3 --features 16 --victim-epochs 10 --final-epochs 14 `
  --output qscout_main_study.csv
```

The runner writes per-run rows and an aggregated mean/std summary CSV. Do not
call a configuration final until the capacity sweep and main-study summaries are
complete.
