# QScout 2.0 CCF-A Strong-Baseline Closure Update

Date: 2026-07-08

## Status

This update addresses the most direct CCF-A reviewer objection from the previous
readiness review: the CyberSecEval add-on was no longer allowed to compare only
against a weak or single classical witness.  The new artifact uses a fair-pool
protocol with six query policies on the same candidate pool:

- Random Search
- Classical Active
- INSEC-style Fixed-Pool Search
- AOT-style Ensemble
- Classical Boundary Witness
- QScout-QBW

## Result

Artifact directory:

```text
paper_artifacts/ccfa_20260708/
```

Primary table:

```text
paper_artifacts/ccfa_20260708/cyberseceval_strong_protocol_tables.md
```

Strongest-baseline comparison on the 120-task stratified CyberSecEval
autocomplete subset:

| Budget | Strongest non-quantum baseline | Baseline ASR | QScout ASR | Absolute gain | Paired 95% CI | Relative gain |
|---:|---|---:|---:|---:|---:|---:|
| 4 | Classical Active | 0.1150 | 0.1550 | +4.00 pp | [+2.59, +5.41] | +34.78% |
| 8 | Classical Active | 0.1550 | 0.1767 | +2.17 pp | [+0.74, +3.59] | +13.98% |

AULC across B=4/8:

| Method | AULC |
|---|---:|
| QScout-QBW | 0.1658 |
| Classical Active | 0.1350 |
| AOT-style Ensemble | 0.1217 |
| Classical Boundary Witness | 0.1192 |
| INSEC-style Fixed-Pool Search | 0.1000 |
| Random Search | 0.0983 |

QScout's AULC gain over the strongest non-quantum baseline is +22.84%.

## Diagnostics Added

The same artifact directory now includes:

- `cyberseceval_cost_efficiency.csv`
- `cyberseceval_language_generalization.csv`
- `cyberseceval_cwe_generalization.csv`
- `cyberseceval_failure_boundary.csv`
- `cyberseceval_strong_diagnostics.md`

The diagnostics show that QScout's low-budget gains concentrate in
JavaScript/PHP/Python/C security contexts, while Java/CPP/Rust are closer to
ties or low-base-rate settings.  This should be written as a boundary of the
method, not as universal dominance.

## Claim Boundary

The supported CCF-A-facing claim is:

> Objective-aligned quantum query learning improves low-budget hard-label
> query selection for code-LLM security extraction under fair-pool strong
> baselines.

The unsupported claims remain:

- raw quantum advantage;
- universal ASR dominance at all budgets;
- end-to-end NISQ hardware advantage.

## Verification

Use:

```powershell
& "D:\ProgramData\py2\python.exe" scripts\verify_ccfa_artifacts.py
& "D:\ProgramData\py2\python.exe" -m unittest tests.test_lightweight_artifacts -v
```

The verifier now checks the CyberSecEval strong-baseline strategy coverage,
positive paired CI lower bounds, and presence of cost/generalization/failure
diagnostic artifacts.
