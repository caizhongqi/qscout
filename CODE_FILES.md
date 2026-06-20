# Runnable Code Files

Open this folder in Visual Studio Code:

```text
C:\Users\lenovo\Documents\论文
```

Use Python interpreter:

```text
D:\ProgramData\py2\python.exe
```

## Main Entry Files

- `run_public_datasets_benchmark.py`  
  Main five-public-dataset experiment: MNIST, FashionMNIST, FordA, Wafer, ElectricDevices.

- `run_hardware_simulation.py`  
  PennyLane finite-shot noisy quantum hardware-style simulation.

- `run_torch_benchmark.py`  
  Synthetic large image/time-series CNN/LSTM benchmark.

- `run_experiment.py`  
  Small LoRA-QEA quick test.

## One-Click PowerShell Scripts

- `RUN_SMOKE_TEST.ps1`  
  Fast MNIST smoke test.

- `RUN_PUBLIC_BENCHMARK.ps1`  
  Full five-public-dataset benchmark.

- `RUN_HARDWARE_SIM.ps1`  
  PennyLane noisy hardware-style simulation.

## Core Quantum Code

- `qlea/quantum.py`  
  Density-matrix simulator, quantum gates, Kraus noise channels.

- `qlea/qnn.py`  
  Improved noise-aware data-reuploading QNN.

- `qlea/pennylane_qnn.py`  
  PennyLane `default.mixed` finite-shot circuit replay.

## Attack Code

- `qlea/attack.py`  
  Hard-label QNN surrogate extraction and LoRA projection attack.

## Dataset and Victim Model Code

- `qlea/public_datasets.py`  
  Public dataset loading for MNIST, FashionMNIST, FordA, Wafer, ElectricDevices.

- `qlea/public_victims.py`  
  PyTorch CNN, MLP, 1D-CNN, LSTM victim models.

- `qlea/torch_victims.py`  
  Synthetic image/time-series victim models.

## Outputs

Results are saved in:

```text
outputs/
```

Important output files:

- `outputs/public_datasets_benchmark_results.csv`
- `outputs/hardware_simulation_results.csv`
- `outputs/public_datasets_smoke_results_20260618.csv`
