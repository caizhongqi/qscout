# QScout 2.0 LLM Security Result Mirror Update

Date: 2026-07-08

## Purpose

This update fills the reviewer-facing organization gap without changing the
verified experimental numbers.  The CCF-A evidence already lived under
`paper_artifacts/`, but security-paper reviewers expect a direct
`results/llm_security` layout, a clear baseline registry, and a short runner
entry point.

## Added

```text
scripts/run_llm_security_probe.py
scripts/export_llm_security_results.py
scripts/verify_llm_security_results.py
baselines/
results/llm_security/
```

## Results Mirror

Generated directory:

```text
results/llm_security/
```

Files:

- `summary.csv`
- `budget_summary.csv`
- `strongest_baseline_comparison.csv`
- `aulc_summary.csv`
- `traces/cyberseceval.jsonl`
- `traces/llmseceval.jsonl`
- `traces/securityeval.jsonl`
- `oracle_audit/`

Manifest:

```text
summary_rows=87
seed_rows=435
budget_rows=87
strongest_baseline_rows=14
trace rows:
  cyberseceval.jsonl: 7200
  llmseceval.jsonl: 29250
  securityeval.jsonl: 21780
```

## Baselines

The `baselines/` directory maps paper names to strategy ids:

- Random Search -> `fair_random_comment`
- Risk Prior -> `fair_risk_prior_comment`
- INSEC-style Fixed-Pool Search -> `insec_fixed_pool_comment`
- Classical Active -> `classical_active_comment`
- AOT-style Ensemble -> `aot_ensemble_fixed_pool_comment`
- Classical Boundary Witness -> `classical_boundary_witness_comment`
- QScout-QBW -> `qscout_qbw_comment`

The executable implementations remain centralized in
`qlea/code_completion_attack/benchmark.py` so all methods share the same
candidate pool, detector, and query accounting.

## Claim Boundary

The mirror does not fabricate B=10/20/40 results.  Current committed protocol
budgets are B=4/8/16 for SecurityEval and LLMSecEval, and B=4/8 for
CyberSecEval.  B=10/20/40 requires a separate run.

## Verification

```powershell
& "D:\ProgramData\py2\python.exe" scripts\verify_llm_security_results.py
& "D:\ProgramData\py2\python.exe" scripts\verify_ccfa_artifacts.py
```

Current output:

```text
LLM security results mirror verification passed.
summary_rows=87
strongest_rows=14
trace_counts={cyberseceval: 7200, llmseceval: 29250, securityeval: 21780}

CCF-A artifact verification passed.
main_rows=12 seed_rows=375 mechanism_files=4 cyber_rows=2 trace_main_rows=51030 trace_cyber_rows=7200 oracle_rows=58230 results_summary_rows=87
```
