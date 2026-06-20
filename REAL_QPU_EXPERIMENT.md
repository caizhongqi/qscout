# Real IBM Quantum Experiment

The existing PennyLane experiment is a finite-shot noisy simulation. A physical
IBM result requires an IBM Quantum account, a user-owned token, an operational
backend, and hardware allocation.

```powershell
& "D:\ProgramData\py2\python.exe" -m pip install qiskit qiskit-ibm-runtime
$env:QISKIT_IBM_TOKEN = "your-token-from-IBM-Quantum"
& "D:\ProgramData\py2\python.exe" run_ibm_qpu_experiment.py --samples 32 --shots 2048
```

The runner trains the small controlled NQSE surrogate locally, transpiles its
four-qubit circuit to a physical IBM backend, samples one circuit per held-out
input, reconstructs the `Z/ZZ` feature vector from measurement counts, and saves
physical-QPU agreement to `outputs/ibm_qpu_results.csv`.

Never add the token to source code or commit it. A paper-grade physical-hardware
section requires repeated jobs across multiple calibration dates, backend
metadata, multiple seeds, and a mitigated-versus-unmitigated comparison.
