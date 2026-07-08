# QScout 2.0 CCF-A Artifact Manifest

This folder contains the table-level evidence used by the CCF-A strengthening run on 2026-07-07.  Large completion traces and local model caches remain under `outputs/` and are not committed by default; the CSV files here are the compact reproducibility layer for the manuscript tables.

## Main Results

| File | Purpose |
|---|---|
| `main_ccfa_protocol_tables.md` | Markdown version of the final combined results. |
| `main_strongest_baseline_comparison.csv` | QScout-QBW vs the strongest non-quantum baseline for each dataset, victim, and budget. |
| `main_budget_summary.csv` | Per-method budget summaries with confidence intervals. |
| `main_aulc_summary.csv` | Area-under-low-budget-curve summaries. |
| `main_seed_rows.csv` | Per-seed rows used for paired comparisons. |

## Mechanism Evidence

| File | Setting |
|---|---|
| `mechanism_securityeval_qwen05.csv` | SecurityEval / Qwen2.5-Coder-0.5B |
| `mechanism_llmseceval_qwen05.csv` | LLMSecEval / Qwen2.5-Coder-0.5B |
| `mechanism_securityeval_qwen15.csv` | SecurityEval / Qwen2.5-Coder-1.5B |
| `mechanism_llmseceval_qwen15.csv` | LLMSecEval / Qwen2.5-Coder-1.5B |

These files compare raw `qbw_score` against objective-aligned `actual_qbw_acquisition`.  The intended claim is not that raw quantum uncertainty is sufficient; the evidence supports objective-aligned quantum query learning.

## Ablation and Boundary Evidence

| File | Purpose |
|---|---|
| `strict_qbw_ablation_budget_summary.csv` | Strict QBW ablation with priority/objective gates disabled. |

The strict ablation is a boundary result: naked QBW is weaker than the full objective-aligned QScout-QBW policy.  This should be reported as a limitation and mechanism clarification, not hidden.

## Verification

Run:

```bash
python scripts/verify_ccfa_artifacts.py
```

Expected output:

```text
CCF-A artifact verification passed.
```
