# QScout 2.0 Full-Strong Baseline Update

Date: 2026-07-08

## What Changed

The main CCF-A artifact now includes a full strong-baseline expansion for the
second-victim LLMSecEval setting:

```text
LLMSecEval / Qwen/Qwen2.5-Coder-1.5B-Instruct / B=4,8,16 / 5 seeds
```

The previous artifact compared only:

- Classical Boundary Witness
- QScout-QBW

The refreshed artifact compares:

- Random Search
- Risk Prior
- Classical Active
- INSEC-style Fixed-Pool Search
- AOT-style Ensemble
- Classical Boundary Witness
- QScout-QBW

## Artifact

```text
paper_artifacts/ccfa_20260707/
```

The main seed table increased from 300 rows to 375 rows.

## Result

The strongest non-quantum baseline remains Classical Boundary Witness.

| Budget | Strongest baseline | Baseline ASR | QScout ASR | Abs. gain | Paired 95% CI | Q@Success reduction |
|---:|---|---:|---:|---:|---:|---:|
| 4 | Classical Boundary Witness | 0.7920 | 0.8267 | +3.47 pp | [+2.51, +4.43] | +17.34% |
| 8 | Classical Boundary Witness | 0.8907 | 0.9147 | +2.40 pp | [+1.73, +3.07] | +17.15% |
| 16 | Classical Boundary Witness | 0.9280 | 0.9333 | +0.53 pp | [-0.23, +1.30] | +25.57% |

The correct interpretation is:

- B=4 and B=8 provide statistically positive ASR gains under full strong
  baseline coverage.
- B=16 is a near-saturation setting; its ASR gain is small and the paired CI
  crosses zero, but QScout still reduces Q@Success by 25.57%.
- This strengthens the CCF-A borderline case because the second victim is now
  covered by strong baselines rather than a two-method comparison.

## Verification

```powershell
& "D:\ProgramData\py2\python.exe" scripts\verify_ccfa_artifacts.py
& "D:\ProgramData\py2\python.exe" -m unittest tests.test_lightweight_artifacts -v
```

The verifier checks the 375-row seed artifact and the seven-strategy
LLMSecEval/Qwen1.5 coverage at all three budgets.
