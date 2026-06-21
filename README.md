# QScout

Python source code for variational quantum-circuit models, noisy density-matrix
simulation, public-dataset loaders, classical victim models, and benchmark
runners.

## Setup

```powershell
& "D:\ProgramData\py2\python.exe" -m pip install -r requirements.txt
```

## Run

```powershell
& "D:\ProgramData\py2\python.exe" run.py --mode smoke
```

Use `run.py --help` for available study and figure-generation commands.

## Layout

- `qlea/`: quantum simulator, QNNs, query selection, datasets, and victims.
- `run.py`: unified project entry point.
- `run_active_hardlabel_benchmark.py`: benchmark implementation.
- `run_study_matrix.py`: study-matrix runner.
- `generate_*.py`: figure generation utilities.

## License

MIT.
