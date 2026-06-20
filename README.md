# QScout: Query-Guided Noisy Quantum Surrogate for Hard-Label Extraction

This repository is a research prototype for evaluating whether a physically
valid variational quantum circuit (VQC) can guide query-efficient hard-label
extraction against classical neural networks. A low-rank/LoRA target is included
as a focused parameter-recovery setting; image and time-series victims provide
the broader behavioral-extraction evaluation.

The local environment intentionally uses only NumPy/SciPy/scikit-learn. The
quantum layer is implemented as a small density-matrix simulator so that noisy
NISQ channels such as phase flip and amplitude damping can be tested without
Qiskit or PennyLane.

## Research Question

Given black-box hard-label access to a classical target, can a compact noisy QNN
serve as a query-selection surrogate and improve a final classical clone under a
matched query budget? In the LoRA setting, can that surrogate also support a
rank-constrained estimate of a private low-rank update?

This is not a claim that measurement collapse reveals private training data, nor
that a QNN has a general advantage over classical active learning. The prototype
tests a narrower, falsifiable hypothesis:

> Low-rank adaptation creates a small attack surface that can be approximated by
> few-qubit quantum feature maps under realistic noise.

The complete experimental protocol and reporting gates are in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Current pilot results are explicitly
not sufficient for a top-conference quantum-advantage claim.

## Why This Direction

- QGAN-only inversion is currently too speculative for hard-label, zero-shot
  recovery of training data.
- NISQ-aware circuits are important, but stronger as an experimental section
  than as the whole paper.
- LoRA extraction is timely and aligns well with current qubit limits because
  the object being stolen is already low-rank.
- Stealth via entanglement is not physically meaningful against a classical
  black-box API unless the server accepts coherent quantum queries.

## Run

```powershell
& "D:\ProgramData\py2\python.exe" run.py --mode smoke
```

`run.py` is the project entry point. Use `--mode main`, `--mode ablations`,
`--mode noise`, or `--mode defense` for a study family; append `--figures` to
render evidence-bound figures from completed multi-seed CSVs.

Or run the multi-dataset benchmark:

```powershell
python run_benchmark.py
```

For the larger PyTorch CNN/LSTM benchmark, use your PyTorch interpreter:

```powershell
& "D:\ProgramData\py2\python.exe" run_torch_benchmark.py
```

For the five-public-dataset benchmark:

```powershell
& "D:\ProgramData\py2\python.exe" run_public_datasets_benchmark.py
```

For PennyLane hardware-style noisy quantum simulation:

```powershell
& "D:\ProgramData\py2\python.exe" run_hardware_simulation.py
```

In Visual Studio Code, open this folder and use `Run and Debug`:

- `Run LoRA-QEA Quick Experiment`
- `Run Multi-Dataset Benchmark`
- `Run PyTorch CNN/LSTM Benchmark`
- `Run Five Public Datasets Benchmark`
- `Run PennyLane Hardware Simulation`

The script prints victim accuracy, extraction accuracy, LoRA update cosine
similarity, relative reconstruction error, and results under several noise
levels.

Benchmark outputs are written to `outputs/`:

- `benchmark_results.csv`: all metrics.
- `digits_8x8_images_samples.png`: image samples from the handwritten digit dataset.
- `synthetic_time_series_samples.png`: generated time-series samples.

## Current Victim Models

The current implemented victims are classical models:

- `low_rank_lora_head`: public base linear classifier plus a private low-rank
  LoRA update. This is the only model where parameter recovery is meaningful.
- `shallow_mlp_tabular_low_rank` and `deep_mlp_tabular_low_rank`: MLP victims on
  synthetic tabular data.
- `shallow_mlp_digits_8x8_images` and `deep_mlp_digits_8x8_images`: MLP victims
  on handwritten 8x8 image data.
- `shallow_mlp_synthetic_time_series` and `deep_mlp_synthetic_time_series`: MLP
  victims on generated sequence data.
- `cnn_shape_images_16x16`: PyTorch CNN victim trained on 6000 synthetic images.
- `cnn1d_long_time_series`: PyTorch 1D-CNN victim trained on 8000 time-series samples.
- `lstm_long_time_series`: PyTorch LSTM victim trained on 8000 time-series samples.

The attack is a hard-label black-box extraction attack. For non-LoRA MLP
victims, it measures behavior stealing through victim/QNN agreement. For the
LoRA victim, it additionally reports low-rank update reconstruction metrics.

CNN and LSTM victims are implemented in `run_torch_benchmark.py`. Transformer
and real LoRA-adapted language-model layers are the next expansion step.

## Improved QNN Attack Module

The quantum implementation is in `qlea/quantum.py` and `qlea/qnn.py`.

Theory and limits are documented in `theory_foundation.md`.  Run
`run_theory_sanity.py` to reproduce the conservative information, finite-query,
local-noise, and finite-shot bounds used to frame the experiments.

The complete pre-submission manuscript, including the actual recorded smoke
results and diagrams, is `NQSE_submission_draft.md`. It is intentionally scoped
as a capability-boundary study; it does not claim a universal quantum advantage.

Compared with the initial baseline VQC, the current QNN adds:

- Trainable `RX/RY/RZ` rotations.
- Data re-uploading at every variational layer.
- Configurable entanglement topology: `none`, `linear`, `circular`, `star`, and `full`.
- Multi-observable readout: single-qubit `Z` expectations and neighboring `ZZ` correlations.
- Density-matrix noise simulation: phase flip, bit flip, and amplitude damping.

The current quantum contribution is a noise-aware data-reuploading QNN surrogate
for hard-label extraction attacks against classical neural networks.

`run_hardware_simulation.py` replays the trained QNN circuit on PennyLane
`default.mixed` with finite shots and noise channels. This gives a hardware-like
simulation path before moving to IBM/Qiskit backends.

## Five Public Datasets

`run_public_datasets_benchmark.py` uses five public datasets:

- `MNIST`: 60,000 train / 10,000 test image samples.
- `FashionMNIST`: 60,000 train / 10,000 test image samples.
- `FordA`: UCR time-series dataset.
- `Wafer`: UCR time-series dataset.
- `ElectricDevices`: UCR time-series dataset.

Victim models:

- Images: CNN and MLP.
- Time series: 1D-CNN and LSTM.

The QNN attacker uses 1024 hard-label queries by default and saves metrics to
`outputs/public_datasets_benchmark_results.csv`.

## Repository Layout

- `qlea/quantum.py`: density-matrix simulator, gates, noise channels.
- `qlea/qnn.py`: VQC quantum feature extractor and SPSA training loop.
- `qlea/target.py`: classical LoRA-adapted target model.
- `qlea/datasets.py`: tabular, image, and time-series datasets.
- `qlea/models.py`: classical victim model wrappers.
- `qlea/attack.py`: black-box hard-label extraction and low-rank projection.
- `run.py`: primary entry point for the complete study and figure workflow.
- `run_experiment.py`: legacy end-to-end LoRA toy experiment.
- `run_benchmark.py`: multi-model, multi-dataset benchmark.
- `run_torch_benchmark.py`: larger PyTorch image/time-series benchmark.
- `run_public_datasets_benchmark.py`: five-public-dataset benchmark.
- `run_hardware_simulation.py`: PennyLane finite-shot noisy quantum simulation.
- `research_notes.md`: Chinese literature/positioning notes and direction choice.

## GitHub Baseline Survey

Relevant open-source starting points found during survey:

- `n-azimi/QShield`: hybrid quantum-classical adversarial robustness architecture.
- `Sinestro38/qosf-qgan`: QGAN implementation lineage useful for generator ideas.

This prototype borrows the research framing from those directions but replaces
the dependency-heavy implementation with a compact simulator tailored to LoRA
extraction.

## Real Quantum Hardware

`run_ibm_qpu_experiment.py` is a real IBM QPU replay runner. It needs an IBM
token and an operational physical backend, and does not create a hardware result
until it is actually submitted. See `REAL_QPU_EXPERIMENT.md`.
