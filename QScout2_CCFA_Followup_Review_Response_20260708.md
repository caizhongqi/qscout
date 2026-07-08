# QScout 2.0 Follow-Up Review Response

Date: 2026-07-08

## Judgment

The follow-up review was useful, but it was based on commit
`8d2008703bac8301cbda8ecebb026114ee5e0128`.  The evidence line has since moved
past that commit; the immediate pre-oracle-audit remote HEAD was:

```text
4dc41d9f3b3829e284de9a52bdf1a460ac25150a
```

Several criticisms were already stale:

- `main_seed_rows.csv` is no longer 300 rows; it is 375 rows.
- LLMSecEval/Qwen1.5 is no longer a two-method comparison; it includes seven
  strategies at B=4/8/16.
- CyberSecEval now has a six-method strong-baseline fair-pool table and
  diagnostics.

The review was still correct on two hard points: the artifact needed a stronger
trace-to-table verification chain, and the detector oracle needed explicit
boundary diagnostics.

## Implemented Fix

Added a compact trace artifact:

```text
paper_artifacts/ccfa_trace_20260708/
```

It includes:

- `main_task_outcomes.csv`
- `cyberseceval_task_outcomes.csv`
- `source_file_hashes.csv`
- `completion_cache_manifest.csv`
- `trace_manifest.json`
- `TRACE_MANIFEST.md`

The trace artifact commits detector/task outcomes and SHA-256 manifests for the
local raw CSV/cache files.  It intentionally does not commit the full generated
code payloads because the local generated-code files are hundreds of megabytes.

Added an oracle boundary audit:

```text
paper_artifacts/ccfa_oracle_audit_20260708/
```

It audits 58,230 task-outcome rows and separates effective
unsafe-and-functional successes from vulnerable-but-nonfunctional and
functional-but-not-vulnerable boundary cases.

Dataset-level audit summary:

| Artifact | Dataset | Task rows | Effective rate | Vulnerable nonfunctional | Functional not vulnerable |
|---|---|---:|---:|---:|---:|
| cyberseceval | CyberSecEval | 7,200 | 0.1233 | 0.0082 | 0.6317 |
| main | LLMSecEval | 29,250 | 0.7204 | 0.0000 | 0.2670 |
| main | SecurityEval | 21,780 | 0.8108 | 0.0000 | 0.1865 |

## Verification

New scripts:

```text
scripts/build_trace_artifact.py
scripts/verify_trace_artifact.py
scripts/build_oracle_audit.py
scripts/verify_oracle_audit.py
```

`scripts/verify_ccfa_artifacts.py` now calls both the trace verifier and the
oracle audit verifier.  Current local verification:

```text
CCF-A artifact verification passed.
main_rows=12 seed_rows=375 mechanism_files=4 cyber_rows=2 trace_main_rows=51030 trace_cyber_rows=7200 oracle_rows=58230
```

Standalone trace verification:

```text
CCF-A trace artifact verification passed.
main seed rows verified: 375
CyberSecEval seed rows verified: 60
max_abs_error: 0.0
```

Standalone oracle verification:

```text
CCF-A oracle audit verification passed.
task_outcome_rows: 58230
dataset_rows_verified: 3
boundary_queue_rows: 500
max_abs_error: 0.0
```

## Remaining Claim Boundary

The following review points remain scientifically valid and should be handled in
the manuscript rather than hidden:

- Do not claim query-efficiency improvement for every row; some Q@Success rows
  are negative or saturated.
- Report LLMSecEval/Qwen1.5/B16 as a saturated setting where ASR gain is small
  and CI crosses zero, while Q@Success still improves.
- Do not claim raw QBW alone is sufficient; the supported mechanism is
  objective-aligned quantum query learning.
- Pattern/functionality detectors are still weaker than full human/static/unit
  test validation; the new oracle audit improves transparency but is not a
  substitute for human/static/unit-test exploit validation.
