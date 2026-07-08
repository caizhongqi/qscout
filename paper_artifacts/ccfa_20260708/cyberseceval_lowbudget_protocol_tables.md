# CCF-A Fair-Pool Protocol Tables

## Protocol

- All fixed-pool baselines rank or sample from the same candidate prompt/code mutation pool.
- Main metric: Unsafe-and-Functional@Q.
- Main comparison: QScout-QBW vs the strongest non-quantum baseline available in the same setting.
- Gate: +5 pp Unsafe-and-Functional@Q, or >=20% Q@Success reduction, or >=10% AULC gain.

## Budget Summary

| Dataset | Model | Budget | Method | Unsafe-and-Functional@Q | 95% CI | Q@Success | Queries/task | Seeds |
|---|---|---:|---|---:|---:|---:|---:|---|
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | Classical Boundary Witness | 0.0717 | +/- 0.0040 | 1.81 | 3.84 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | QScout-QBW | 0.1300 | +/- 0.0083 | 2.13 | 3.76 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | Classical Boundary Witness | 0.1433 | +/- 0.0061 | 4.03 | 7.43 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | QScout-QBW | 0.1500 | +/- 0.0000 | 2.59 | 7.19 | 7,19,31,43,59 |

## AULC

| Dataset | Model | Method | Budgets | AULC |
|---|---|---|---|---:|
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | Classical Boundary Witness | 4,8 | 0.1075 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | QScout-QBW | 4,8 | 0.1400 |

## Strongest Non-Quantum Baseline Comparison

| Dataset | Model | Budget | Main | Strongest baseline | Abs. gain | Paired 95% CI | Rel. gain | Q@Success reduction |
|---|---|---:|---|---|---:|---:|---:|---:|
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | QScout-QBW | Classical Boundary Witness | +5.83 pp | [+5.10, +6.56] | +81.40% | -18.13% |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | QScout-QBW | Classical Boundary Witness | +0.67 pp | [+0.06, +1.28] | +4.65% | +35.78% |

## Gate Summary

| Dataset | Model | Budget | Effect gate | Strict CI gate | AULC gain |
|---|---|---:|---:|---:|---:|
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | 1 | 1 | +30.23% |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | 1 | 1 | +30.23% |
