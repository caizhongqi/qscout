# QScout 2.0 CCF-A Oracle Audit Update

Date: 2026-07-08

## Purpose

This update addresses a concrete CCF-A review risk: `Unsafe-and-Functional@Q`
should not be trusted as a single opaque detector number.  We therefore added a
trace-level oracle boundary audit that separates:

- effective unsafe-and-functional successes;
- vulnerable but non-functional outputs;
- functional but non-vulnerable outputs;
- neither-vulnerable-nor-functional failures.

## Artifact

```text
paper_artifacts/ccfa_oracle_audit_20260708/
```

Files:

- `oracle_boundary_summary.csv`
- `oracle_boundary_by_dataset.csv`
- `oracle_boundary_by_cwe.csv`
- `oracle_boundary_by_language.csv`
- `oracle_boundary_queue.csv`
- `oracle_audit_manifest.json`
- `ORACLE_AUDIT.md`

## Result

| Artifact | Dataset | Task rows | Effective rate | Vulnerable nonfunctional | Functional not vulnerable |
|---|---|---:|---:|---:|---:|
| cyberseceval | CyberSecEval | 7,200 | 0.1233 | 0.0082 | 0.6317 |
| main | LLMSecEval | 29,250 | 0.7204 | 0.0000 | 0.2670 |
| main | SecurityEval | 21,780 | 0.8108 | 0.0000 | 0.1865 |

Interpretation:

- SecurityEval and LLMSecEval show no vulnerable-but-nonfunctional boundary
  mass under the committed task-level outcomes.
- CyberSecEval is much harder: only 12.33% of task outcomes are effective, and
  63.17% are functional but not vulnerable.  This explains why CyberSecEval ASR
  is much lower than SecurityEval/LLMSecEval and should be described as a harder
  external benchmark, not a failed reproduction.
- The 500-row `oracle_boundary_queue.csv` is a hash-only queue for future
  human/static/unit-test validation.

## Verification

```powershell
& "D:\ProgramData\py2\python.exe" scripts\verify_oracle_audit.py
& "D:\ProgramData\py2\python.exe" scripts\verify_ccfa_artifacts.py
```

Current output:

```text
CCF-A oracle audit verification passed.
task_outcome_rows: 58230
dataset_rows_verified: 3
boundary_queue_rows: 500
max_abs_error: 0.0

CCF-A artifact verification passed.
main_rows=12 seed_rows=375 mechanism_files=4 cyber_rows=2 trace_main_rows=51030 trace_cyber_rows=7200 oracle_rows=58230
```

## Claim Boundary

This audit improves detector transparency, but it is not a substitute for
human-level exploit validation, static-analysis validation, or unit-test
execution.  The manuscript should state this boundary explicitly.
