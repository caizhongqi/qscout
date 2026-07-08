# QScout-QBW: Objective-Aligned Quantum Boundary Witnessing for Hard-Label Code-LLM Security Probing

## Abstract

Hard-label code-LLM security probing is a query-selection problem: an attacker
or evaluator sends prompts to a black-box code model and observes only generated
code plus a binary security/functionality verdict.  Under a small query budget,
the central challenge is not training a surrogate classifier, but deciding which
query variants are most likely to expose unsafe-and-functional completions.

We present QScout-QBW, an objective-aligned quantum query learning framework for
hard-label code-LLM security probing.  QScout-QBW constructs a fair candidate
pool shared with all baselines, represents candidates through a quantum
boundary-witness layer, and aligns quantum boundary evidence with observed
hard-label utility.  The supported claim is not hardware-level or
complexity-theoretic quantum advantage; rather, QScout-QBW provides empirical
low-budget query-selection gains when quantum boundary signals are guarded by
objective alignment and detector feedback.

Across SecurityEval, LLMSecEval, and a stratified CyberSecEval/PurpleLlama
autocomplete subset, QScout-QBW is evaluated against Random Search, Risk Prior,
INSEC-style Fixed-Pool Search, Classical Active, AOT-style Ensemble, and
Classical Boundary Witness baselines.  The committed evidence package covers
two Qwen2.5-Coder victim models, five seeds, low-query budgets, trace
reaggregation, mechanism diagnostics, strict ablation, and oracle boundary
audit.

## Main Claim

The paper should make the following claim:

> Objective-aligned quantum boundary witnessing improves low-budget hard-label
> query selection for code-LLM security probing over strong non-quantum
> fair-pool baselines.

The paper should not claim:

- hardware-level quantum advantage;
- complexity-theoretic quantum speedup;
- raw quantum boundary witness alone is sufficient;
- universal query-efficiency improvement in every setting;
- human-level exploit validation from pattern/functionality detectors.

## Problem Setting

### Threat Model

The evaluator has black-box access to a code LLM.  For each security task, the
evaluator can submit a limited number of prompt variants and receives generated
code.  A hard-label oracle marks each generated completion as:

```text
vulnerable
functional
unsafe-and-functional
```

The objective is to maximize Unsafe-and-Functional@Q under a small query budget
Q.

### Query Learning View

The core problem is not model extraction in the classical surrogate-learning
sense.  It is hard-label query selection:

```text
candidate prompt variants
        -> query selector
        -> black-box code LLM
        -> generated code
        -> hard-label security/functionality oracle
        -> update query selector
```

## Method

QScout-QBW has four layers:

1. Fair candidate pool shared by all methods.
2. Classical risk, lexical, detector, and boundary controls.
3. Quantum boundary-witness scoring with density/fidelity-style boundary
   evidence.
4. Objective-alignment layer that calibrates quantum signals using observed
   hard-label utility.

This design is necessary because the strict ablation shows that naked quantum
boundary witness scores are not sufficient.  The empirical contribution comes
from the objective-aligned quantum acquisition function.

## Baselines

The baseline registry is committed under:

```text
baselines/
```

The compared strategies are:

| Paper name | Strategy id | Quantum |
|---|---|---:|
| Random Search | `fair_random_comment` | no |
| Risk Prior | `fair_risk_prior_comment` | no |
| INSEC-style Fixed-Pool Search | `insec_fixed_pool_comment` | no |
| Classical Active | `classical_active_comment` | no |
| AOT-style Ensemble | `aot_ensemble_fixed_pool_comment` | no |
| Classical Boundary Witness | `classical_boundary_witness_comment` | no |
| QScout-QBW | `qscout_qbw_comment` | yes |

## Experimental Evidence

### Main Artifact

```text
paper_artifacts/ccfa_20260707/
```

The main artifact contains 375 seed-level rows across:

- SecurityEval and LLMSecEval;
- Qwen2.5-Coder-0.5B-Instruct and Qwen2.5-Coder-1.5B-Instruct;
- budgets B=4/8/16;
- five seeds 7/19/31/43/59.

### CyberSecEval Add-On

```text
paper_artifacts/ccfa_20260708/
```

The CyberSecEval/PurpleLlama autocomplete add-on uses a deterministic
120-task stratified subset covering 8 languages and 50 CWE families.  Under the
same fair-pool protocol, QScout-QBW beats the strongest non-quantum baseline
at B=4 and B=8 with positive paired confidence intervals.

### Results Mirror

```text
results/llm_security/
```

This directory mirrors the paper artifacts in a reviewer-facing layout:

- `summary.csv`
- `budget_summary.csv`
- `strongest_baseline_comparison.csv`
- `aulc_summary.csv`
- `traces/*.jsonl`
- `oracle_audit/`

The mirror does not fabricate B=10/20/40 results.  The current committed
protocol uses B=4/8/16 for SecurityEval and LLMSecEval, and B=4/8 for
CyberSecEval.

## Verification

The full standard-library verification command is:

```powershell
& "D:\ProgramData\py2\python.exe" scripts\verify_ccfa_artifacts.py
```

Current expected output:

```text
CCF-A artifact verification passed.
main_rows=12 seed_rows=375 mechanism_files=4 cyber_rows=2 trace_main_rows=51030 trace_cyber_rows=7200 oracle_rows=58230 results_summary_rows=87
```

The verifier checks:

- table-level evidence;
- five-seed coverage;
- strong baseline coverage;
- trace-to-table reaggregation;
- oracle boundary audit;
- `results/llm_security` mirror consistency.

## Limitations

1. The work does not prove quantum advantage in the hardware or complexity
   sense.
2. QScout-QBW should be described as objective-aligned quantum query learning,
   not raw QBW.
3. LLMSecEval/Qwen1.5/B16 is a saturated setting: ASR gain is small and the CI
   crosses zero, although Q@Success improves.
4. The oracle audit improves transparency but does not replace human review,
   static analysis, or unit-test validation.
5. Current external validity is limited to Qwen2.5-Coder victim models; a
   stronger paper should add DeepSeek-Coder, CodeLlama, or StarCoder-family
   victims.

## Next Experimental Extension

The next run should add a different code-model family:

```text
SecurityEval + LLMSecEval
DeepSeek-Coder-1.3B or CodeLlama-7B
B=4/8/16 or B=10/20/40
5 seeds
same seven strategies
```

Acceptance criterion for the extension:

```text
QScout-QBW beats the strongest non-quantum baseline by
  >= 5 pp UnsafeFunctional@Q, or
  >= 20% Q@Success reduction, or
  >= 10% AULC gain,
with paired CI not crossing zero in the low-budget regime.
```
