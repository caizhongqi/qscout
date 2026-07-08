# CCF-A Fair-Pool Protocol Tables

## Protocol

- All fixed-pool baselines rank or sample from the same candidate prompt/code mutation pool.
- Main metric: Unsafe-and-Functional@Q.
- Main comparison: QScout-QBW vs the strongest non-quantum baseline available in the same setting.
- Gate: +5 pp Unsafe-and-Functional@Q, or >=20% Q@Success reduction, or >=10% AULC gain.

## Budget Summary

| Dataset | Model | Budget | Method | Unsafe-and-Functional@Q | 95% CI | Q@Success | Queries/task | Seeds |
|---|---|---:|---|---:|---:|---:|---:|---|
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | AOT-style Ensemble | 0.5080 | +/- 0.0096 | 2.10 | 3.04 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | Classical Boundary Witness | 0.7133 | +/- 0.0109 | 2.00 | 2.57 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | Random Search | 0.4920 | +/- 0.0412 | 1.99 | 3.02 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | Risk Prior | 0.4573 | +/- 0.0032 | 2.01 | 3.09 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | INSEC-style Fixed-Pool Search | 0.4627 | +/- 0.0067 | 2.10 | 3.12 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | QScout-QBW | 0.9000 | +/- 0.0000 | 1.48 | 1.73 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | AOT-style Ensemble | 0.6040 | +/- 0.0032 | 2.73 | 4.82 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | Classical Boundary Witness | 0.8320 | +/- 0.0157 | 2.63 | 3.54 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | Random Search | 0.6787 | +/- 0.0140 | 3.02 | 4.62 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | Risk Prior | 0.5880 | +/- 0.0064 | 2.99 | 5.06 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | INSEC-style Fixed-Pool Search | 0.6187 | +/- 0.0026 | 3.17 | 5.01 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | QScout-QBW | 0.9493 | +/- 0.0032 | 1.72 | 2.04 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | AOT-style Ensemble | 0.7253 | +/- 0.0049 | 4.24 | 7.47 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | Classical Boundary Witness | 0.9707 | +/- 0.0032 | 3.35 | 3.72 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | Random Search | 0.8360 | +/- 0.0221 | 5.14 | 6.92 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | Risk Prior | 0.6507 | +/- 0.0032 | 3.67 | 7.98 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | INSEC-style Fixed-Pool Search | 0.6867 | +/- 0.0000 | 4.13 | 7.85 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | QScout-QBW | 0.9800 | +/- 0.0000 | 2.02 | 2.30 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | AOT-style Ensemble | 0.5533 | +/- 0.0041 | 1.82 | 2.79 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | Classical Active | 0.6040 | +/- 0.0067 | 1.71 | 2.61 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | Classical Boundary Witness | 0.7920 | +/- 0.0096 | 1.72 | 2.19 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | Random Search | 0.5893 | +/- 0.0243 | 1.79 | 2.70 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | Risk Prior | 0.5533 | +/- 0.0000 | 1.68 | 2.72 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | INSEC-style Fixed-Pool Search | 0.5040 | +/- 0.0032 | 1.69 | 2.84 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | QScout-QBW | 0.8267 | +/- 0.0000 | 1.42 | 1.87 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | AOT-style Ensemble | 0.6427 | +/- 0.0067 | 2.49 | 4.46 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | Classical Active | 0.8040 | +/- 0.0114 | 2.84 | 3.85 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | Classical Boundary Witness | 0.8907 | +/- 0.0052 | 2.18 | 2.81 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | Random Search | 0.7120 | +/- 0.0444 | 2.47 | 4.06 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | Risk Prior | 0.6440 | +/- 0.0052 | 2.35 | 4.36 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | INSEC-style Fixed-Pool Search | 0.6547 | +/- 0.0049 | 2.73 | 4.55 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | QScout-QBW | 0.9147 | +/- 0.0026 | 1.80 | 2.33 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | AOT-style Ensemble | 0.7413 | +/- 0.0076 | 3.65 | 6.85 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | Classical Active | 0.9147 | +/- 0.0064 | 3.77 | 4.82 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | Classical Boundary Witness | 0.9280 | +/- 0.0076 | 2.68 | 3.64 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | Random Search | 0.8373 | +/- 0.0178 | 4.10 | 6.04 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | Risk Prior | 0.6933 | +/- 0.0000 | 2.99 | 6.98 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | INSEC-style Fixed-Pool Search | 0.7093 | +/- 0.0032 | 3.42 | 7.07 | 7,19,31,43,59 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | QScout-QBW | 0.9333 | +/- 0.0000 | 1.99 | 2.93 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | AOT-style Ensemble | 0.6595 | +/- 0.0061 | 1.31 | 2.23 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | Classical Boundary Witness | 0.7240 | +/- 0.0040 | 1.54 | 2.22 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | Random Search | 0.6860 | +/- 0.0332 | 1.53 | 2.31 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | Risk Prior | 0.6529 | +/- 0.0000 | 1.37 | 2.28 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | INSEC-style Fixed-Pool Search | 0.6314 | +/- 0.0040 | 1.28 | 2.28 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | QScout-QBW | 0.9124 | +/- 0.0040 | 2.01 | 2.18 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | AOT-style Ensemble | 0.7306 | +/- 0.0040 | 1.77 | 3.45 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | Classical Boundary Witness | 0.8331 | +/- 0.0119 | 2.27 | 3.23 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | Random Search | 0.7719 | +/- 0.0249 | 2.06 | 3.42 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | Risk Prior | 0.6992 | +/- 0.0040 | 1.73 | 3.61 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | INSEC-style Fixed-Pool Search | 0.7240 | +/- 0.0040 | 1.93 | 3.61 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | QScout-QBW | 0.9802 | +/- 0.0040 | 2.33 | 2.44 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | AOT-style Ensemble | 0.8446 | +/- 0.0061 | 3.23 | 5.21 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | Classical Boundary Witness | 0.9719 | +/- 0.0121 | 3.33 | 3.69 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | Random Search | 0.9041 | +/- 0.0167 | 3.44 | 4.65 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | Risk Prior | 0.7752 | +/- 0.0032 | 2.83 | 5.79 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | INSEC-style Fixed-Pool Search | 0.8017 | +/- 0.0000 | 2.87 | 5.47 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | QScout-QBW | 1.0000 | +/- 0.0000 | 2.63 | 2.63 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | AOT-style Ensemble | 0.7074 | +/- 0.0040 | 1.62 | 2.32 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | Classical Boundary Witness | 0.7322 | +/- 0.0083 | 1.64 | 2.27 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | Random Search | 0.6810 | +/- 0.0189 | 1.51 | 2.31 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | Risk Prior | 0.6975 | +/- 0.0040 | 1.47 | 2.24 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | INSEC-style Fixed-Pool Search | 0.6380 | +/- 0.0032 | 1.41 | 2.35 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | QScout-QBW | 0.9008 | +/- 0.0072 | 2.02 | 2.22 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | AOT-style Ensemble | 0.8165 | +/- 0.0032 | 2.22 | 3.28 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | Classical Boundary Witness | 0.8645 | +/- 0.0121 | 2.39 | 3.15 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | Random Search | 0.7950 | +/- 0.0188 | 2.16 | 3.36 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | Risk Prior | 0.7934 | +/- 0.0000 | 2.07 | 3.29 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | INSEC-style Fixed-Pool Search | 0.7884 | +/- 0.0040 | 2.38 | 3.57 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | QScout-QBW | 0.9917 | +/- 0.0000 | 2.40 | 2.44 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | AOT-style Ensemble | 0.8760 | +/- 0.0072 | 2.96 | 4.58 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | Classical Boundary Witness | 0.9802 | +/- 0.0040 | 3.04 | 3.29 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | Random Search | 0.9421 | +/- 0.0177 | 3.32 | 4.05 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | Risk Prior | 0.8298 | +/- 0.0065 | 2.52 | 4.81 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | INSEC-style Fixed-Pool Search | 0.8512 | +/- 0.0000 | 3.02 | 4.95 | 7,19,31,43,59 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | QScout-QBW | 1.0000 | +/- 0.0000 | 2.51 | 2.51 | 7,19,31,43,59 |

## AULC

| Dataset | Model | Method | Budgets | AULC |
|---|---|---|---|---:|
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | AOT-style Ensemble | 4,8,16 | 0.6284 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | Classical Boundary Witness | 4,8,16 | 0.8584 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | Random Search | 4,8,16 | 0.7000 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | Risk Prior | 4,8,16 | 0.5871 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | INSEC-style Fixed-Pool Search | 4,8,16 | 0.6153 |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | QScout-QBW | 4,8,16 | 0.9513 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | AOT-style Ensemble | 4,8,16 | 0.6607 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | Classical Active | 4,8,16 | 0.8076 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | Classical Boundary Witness | 4,8,16 | 0.8867 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | Random Search | 4,8,16 | 0.7333 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | Risk Prior | 4,8,16 | 0.6453 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | INSEC-style Fixed-Pool Search | 4,8,16 | 0.6478 |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | QScout-QBW | 4,8,16 | 0.9062 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | AOT-style Ensemble | 4,8,16 | 0.7567 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | Classical Boundary Witness | 4,8,16 | 0.8612 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | Random Search | 4,8,16 | 0.8017 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | Risk Prior | 4,8,16 | 0.7168 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | INSEC-style Fixed-Pool Search | 4,8,16 | 0.7344 |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | QScout-QBW | 4,8,16 | 0.9755 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | AOT-style Ensemble | 4,8,16 | 0.8182 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | Classical Boundary Witness | 4,8,16 | 0.8810 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | Random Search | 4,8,16 | 0.8251 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | Risk Prior | 4,8,16 | 0.7895 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | INSEC-style Fixed-Pool Search | 4,8,16 | 0.7843 |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | QScout-QBW | 4,8,16 | 0.9793 |

## Strongest Non-Quantum Baseline Comparison

| Dataset | Model | Budget | Main | Strongest baseline | Abs. gain | Paired 95% CI | Rel. gain | Q@Success reduction |
|---|---|---:|---|---|---:|---:|---:|---:|
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | QScout-QBW | Classical Boundary Witness | +18.67 pp | [+17.57, +19.76] | +26.17% | +26.00% |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | QScout-QBW | Classical Boundary Witness | +11.73 pp | [+10.00, +13.47] | +14.10% | +34.64% |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | QScout-QBW | Classical Boundary Witness | +0.93 pp | [+0.61, +1.25] | +0.96% | +39.77% |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | QScout-QBW | Classical Boundary Witness | +3.47 pp | [+2.51, +4.43] | +4.38% | +17.34% |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | QScout-QBW | Classical Boundary Witness | +2.40 pp | [+1.73, +3.07] | +2.69% | +17.15% |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | QScout-QBW | Classical Boundary Witness | +0.53 pp | [-0.23, +1.30] | +0.57% | +25.57% |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | QScout-QBW | Classical Boundary Witness | +18.84 pp | [+18.24, +19.45] | +26.03% | -29.94% |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | QScout-QBW | Classical Boundary Witness | +14.71 pp | [+13.77, +15.66] | +17.66% | -2.62% |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | QScout-QBW | Classical Boundary Witness | +2.81 pp | [+1.60, +4.02] | +2.89% | +21.20% |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | QScout-QBW | Classical Boundary Witness | +16.86 pp | [+16.03, +17.69] | +23.02% | -23.66% |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | QScout-QBW | Classical Boundary Witness | +12.73 pp | [+11.52, +13.94] | +14.72% | -0.22% |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | QScout-QBW | Classical Boundary Witness | +1.98 pp | [+1.59, +2.38] | +2.02% | +17.24% |

## Gate Summary

| Dataset | Model | Budget | Effect gate | Strict CI gate | AULC gain |
|---|---|---:|---:|---:|---:|
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | 1 | 1 | +10.82% |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | 1 | 1 | +10.82% |
| llmseceval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | 1 | 1 | +10.82% |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | 0 | 0 | +2.21% |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | 0 | 0 | +2.21% |
| llmseceval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | 1 | 0 | +2.21% |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 4 | 1 | 1 | +13.28% |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 8 | 1 | 1 | +13.28% |
| securityeval | Qwen/Qwen2.5-Coder-0.5B-Instruct | 16 | 1 | 1 | +13.28% |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 4 | 1 | 1 | +11.16% |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 8 | 1 | 1 | +11.16% |
| securityeval | Qwen/Qwen2.5-Coder-1.5B-Instruct | 16 | 1 | 1 | +11.16% |
