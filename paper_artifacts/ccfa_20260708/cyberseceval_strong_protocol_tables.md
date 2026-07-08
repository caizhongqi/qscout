# CCF-A Fair-Pool Protocol Tables

## Protocol

- All fixed-pool baselines rank or sample from the same candidate prompt/code mutation pool.
- Main metric: Unsafe-and-Functional@Q.
- Main comparison: QScout-QBW vs the strongest non-quantum baseline available in the same setting.
- Gate: +5 pp Unsafe-and-Functional@Q, or >=20% Q@Success reduction, or >=10% AULC gain.

## Budget Summary

| Dataset | Model | Budget | Method | Unsafe-and-Functional@Q | 95% CI | Q@Success | Queries/task | Seeds |
|---|---|---:|---|---:|---:|---:|---:|---|
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | AOT-style Ensemble | 0.1067 | +/- 0.0033 | 2.34 | 3.82 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | Classical Active | 0.1150 | +/- 0.0095 | 2.05 | 3.78 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | Classical Boundary Witness | 0.0883 | +/- 0.0040 | 2.22 | 3.84 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | Random Search | 0.0683 | +/- 0.0167 | 1.94 | 3.86 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | INSEC-style Fixed-Pool Search | 0.0750 | +/- 0.0000 | 2.09 | 3.86 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | QScout-QBW | 0.1550 | +/- 0.0083 | 2.08 | 3.70 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | AOT-style Ensemble | 0.1367 | +/- 0.0040 | 3.18 | 7.34 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | Classical Active | 0.1550 | +/- 0.0151 | 3.23 | 7.27 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | Classical Boundary Witness | 0.1500 | +/- 0.0000 | 3.84 | 7.38 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | Random Search | 0.1283 | +/- 0.0083 | 3.74 | 7.45 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | INSEC-style Fixed-Pool Search | 0.1250 | +/- 0.0000 | 4.21 | 7.53 | 7,19,31,43,59 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | QScout-QBW | 0.1767 | +/- 0.0033 | 2.53 | 7.03 | 7,19,31,43,59 |

## AULC

| Dataset | Model | Method | Budgets | AULC |
|---|---|---|---|---:|
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | AOT-style Ensemble | 4,8 | 0.1217 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | Classical Active | 4,8 | 0.1350 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | Classical Boundary Witness | 4,8 | 0.1192 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | Random Search | 4,8 | 0.0983 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | INSEC-style Fixed-Pool Search | 4,8 | 0.1000 |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | QScout-QBW | 4,8 | 0.1658 |

## Strongest Non-Quantum Baseline Comparison

| Dataset | Model | Budget | Main | Strongest baseline | Abs. gain | Paired 95% CI | Rel. gain | Q@Success reduction |
|---|---|---:|---|---|---:|---:|---:|---:|
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | QScout-QBW | Classical Active | +4.00 pp | [+2.59, +5.41] | +34.78% | -1.65% |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | QScout-QBW | Classical Active | +2.17 pp | [+0.74, +3.59] | +13.98% | +21.82% |

## Gate Summary

| Dataset | Model | Budget | Effect gate | Strict CI gate | AULC gain |
|---|---|---:|---:|---:|---:|
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | 1 | 1 | +22.84% |
| cyberseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | 1 | 1 | +22.84% |
