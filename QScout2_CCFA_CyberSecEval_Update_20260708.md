# QScout 2.0 CCF-A Dataset Strengthening: CyberSecEval

## Chinese Conclusion

This round stops treating MNIST/CIFAR-style image datasets as the main evidence
for an LLM security attack.  The CCF-A-facing evidence is strengthened with
Meta PurpleLlama/CyberSecEval autocomplete, a public secure-code benchmark that
matches the paper's code-LLM hard-label extraction threat model.

Implemented scope:

- integrated the CyberSecEval/PurpleLlama autocomplete format;
- preserved official fields:
  `prompt_id/file_path/cwe_identifier/rule/pattern_desc/language`;
- built a deterministic stratified 120-task subset covering 8 languages and 50
  CWE families;
- evaluated Qwen2.5-Coder-0.5B-Instruct with 5 seeds;
- used low-query budgets B=4 and B=8;
- compared QScout-QBW against Random, Classical Active, INSEC-style fixed-pool,
  AOT-style ensemble, and Classical Boundary Witness baselines;
- reported Unsafe-and-Functional@Q, Q@Success, AULC, paired 95% CI, cost,
  language/CWE generalization, and failure-boundary diagnostics.

## English Positioning

This update strengthens QScout's CCF-A evidence by adding a modern public
secure-code benchmark rather than unrelated image classification datasets.  The
CyberSecEval autocomplete tasks match the paper's threat model: a black-box code
LLM is queried with code-completion contexts, and the attacker observes whether
unsafe yet functional code is produced under a limited query budget.

## Literature Rationale

- PurpleLlama/CyberSecEval is a public benchmark line for evaluating LLM
  cybersecurity risks and secure code generation.
- CyberSecEval 2/3/4 broaden the suite beyond secure coding, which makes it a
  recognizable benchmark family for security reviewers.
- CWEval is relevant follow-up work because it argues for outcome-driven
  functional-and-security evaluation; this motivates `Unsafe-and-Functional@Q`
  instead of raw vulnerable suggestion rate.

## Strong-Baseline Result

Clean artifact directory:

```text
paper_artifacts/ccfa_20260708/
```

Main table:

```text
paper_artifacts/ccfa_20260708/cyberseceval_strong_protocol_tables.md
```

Strongest-baseline comparison on the 120-task stratified CyberSecEval
autocomplete subset:

| Budget | Strongest non-quantum baseline | Baseline ASR | QScout ASR | Absolute gain | Paired 95% CI | Relative gain |
|---:|---|---:|---:|---:|---:|---:|
| 4 | Classical Active | 0.1150 | 0.1550 | +4.00 pp | [+2.59, +5.41] | +34.78% |
| 8 | Classical Active | 0.1550 | 0.1767 | +2.17 pp | [+0.74, +3.59] | +13.98% |

Query-efficiency signal:

| Budget | Classical Q@Success | QScout Q@Success | Reduction |
|---:|---:|---:|---:|
| 4 | 2.0470 | 2.0808 | -1.65% |
| 8 | 3.2325 | 2.5273 | +21.82% |

AULC:

| Method | AULC |
|---|---:|
| QScout-QBW | 0.1658 |
| Classical Active | 0.1350 |
| AOT-style Ensemble | 0.1217 |
| Classical Boundary Witness | 0.1192 |
| INSEC-style Fixed-Pool Search | 0.1000 |
| Random Search | 0.0983 |

Interpretation:

- The old single-baseline CyberSecEval result is superseded by the
  strong-baseline fair-pool result above.
- QScout now beats the strongest non-quantum baseline at B=4 and B=8 with
  positive paired 95% CI lower bounds.
- The result supports a low-budget query-learning claim.  It should not be
  written as universal ASR dominance or raw quantum advantage.

## Diagnostics

Additional committed diagnostics:

- `cyberseceval_cost_efficiency.csv`
- `cyberseceval_language_generalization.csv`
- `cyberseceval_cwe_generalization.csv`
- `cyberseceval_failure_boundary.csv`
- `cyberseceval_strong_diagnostics.md`

The diagnostics show strong gains on JavaScript/PHP/Python/C security contexts,
with ties or low-base-rate behavior on Java/CPP/Rust.  These are useful
failure-boundary results for a security conference paper.

## Implemented Code

- `qlea/code_completion_attack/cyberseceval.py`
- `scripts/build_cyberseceval_subset.py`
- `run_llm_topconf_streaming_matrix.py` dataset choice:
  `cyberseceval` / `cyberseceval_autocomplete`
- CyberSecEval-specific QBW low-budget activation and CWE-aligned acquisition
  anchors in `qlea/code_completion_attack/benchmark.py`
- `generate_cyberseceval_diagnostics.py`
- strengthened checks in `scripts/verify_ccfa_artifacts.py`

## Reproduction

Generate the stratified subset:

```powershell
& "D:\ProgramData\py2\python.exe" scripts\build_cyberseceval_subset.py `
  --limit 120 `
  --output data_public\cyberseceval_autocomplete_subset_120.json
```

Run the strong-baseline matrix:

```powershell
& "D:\ProgramData\py2\python.exe" run_llm_topconf_streaming_matrix.py `
  --target hf `
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct `
  --dataset cyberseceval `
  --dataset-path data_public\cyberseceval_autocomplete_subset_120.json `
  --strategies fair_random_comment,classical_active_comment,insec_fixed_pool_comment,aot_ensemble_fixed_pool_comment,classical_boundary_witness_comment,qscout_qbw_comment `
  --budgets 4,8 `
  --seeds 7,19,31,43,59 `
  --prompt-mode raw `
  --max-new-tokens 80 `
  --batch-size 16 `
  --output-dir outputs\ccfa_protocol_cyberseceval_qwen05_120_5seed_strong_v2_20260708
```

Generate tables and diagnostics:

```powershell
& "D:\ProgramData\py2\python.exe" generate_ccfa_protocol_tables.py `
  --roots outputs\ccfa_protocol_cyberseceval_qwen05_120_5seed_strong_v2_20260708 `
  --output-dir outputs\ccfa_protocol_cyberseceval_qwen05_120_5seed_strong_v2_tables_20260708 `
  --main-method qscout_qbw_comment

& "D:\ProgramData\py2\python.exe" generate_cyberseceval_diagnostics.py `
  --root outputs\ccfa_protocol_cyberseceval_qwen05_120_5seed_strong_v2_20260708 `
  --tables outputs\ccfa_protocol_cyberseceval_qwen05_120_5seed_strong_v2_tables_20260708 `
  --output-dir outputs\ccfa_protocol_cyberseceval_qwen05_120_5seed_strong_v2_diagnostics_20260708 `
  --main-method qscout_qbw_comment
```
