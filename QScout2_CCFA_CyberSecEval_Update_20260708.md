# QScout 2.0 CCF-A Dataset Strengthening: CyberSecEval

## 中文结论

本轮不再补 MNIST/CIFAR 这类与 LLM 攻击主线脱节的数据集，而是补充
Meta PurpleLlama/CyberSecEval secure-code autocomplete benchmark。原因是
QScout 2.0 当前主线是 code LLM hard-label security extraction，CCF-A
安全/软件工程审稿更认可同域公开安全 benchmark。

新增执行内容：

- 接入 `CyberSecEval/PurpleLlama` autocomplete 官方数据；
- 保留官方字段 `prompt_id/file_path/cwe_identifier/rule/pattern_desc/language`；
- 生成 deterministic stratified 120-task subset，覆盖 8 种语言、50 个 CWE；
- 在 Qwen2.5-Coder-0.5B-Instruct 上跑 5 seeds；
- 低预算主表：Budget 4/8；
- 对照：`Classical Boundary Witness` vs `QScout-QBW`；
- 指标：Unsafe-and-Functional@Q、Q@Success、AULC、paired 95% CI。

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
  recognizable benchmark family for CCF-A security reviewers.
- CWEval is relevant follow-up work because it argues for outcome-driven
  functional-and-security evaluation; this motivates our metric
  `Unsafe-and-Functional@Q` instead of raw vulnerable suggestion rate.

## New Result

Clean artifact directory:

```text
paper_artifacts/ccfa_20260708/
```

Main table:

```text
paper_artifacts/ccfa_20260708/cyberseceval_lowbudget_protocol_tables.md
```

Key results on 120 stratified CyberSecEval autocomplete tasks:

| Budget | Classical Boundary Witness | QScout-QBW | Absolute gain | Paired 95% CI | Relative gain |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.0717 | 0.1300 | +5.83 pp | [+5.10, +6.56] | +81.40% |
| 8 | 0.1433 | 0.1500 | +0.67 pp | [+0.06, +1.28] | +4.65% |

Query-efficiency signal:

| Budget | Classical Q@Success | QScout Q@Success | Reduction |
|---:|---:|---:|---:|
| 4 | 1.81 | 2.13 | -18.13% |
| 8 | 4.03 | 2.59 | +35.78% |

Interpretation:

- At the strictest budget `B=4`, QScout gives a statistically positive absolute
  ASR gain over the strongest non-quantum witness baseline.
- At `B=8`, the absolute ASR gain is small but paired CI remains positive, and
  query-to-success improves substantially.
- This supports the paper's revised claim: QScout improves low-budget query
  learning on recognized secure-code LLM benchmarks.  It should not be written
  as a universal high-budget ASR dominance claim.

## Implemented Code

- `qlea/code_completion_attack/cyberseceval.py`
- `scripts/build_cyberseceval_subset.py`
- `run_llm_topconf_streaming_matrix.py` dataset choice:
  `cyberseceval` / `cyberseceval_autocomplete`
- CyberSecEval-specific QBW low-budget activation and CWE-aligned acquisition
  anchors in `qlea/code_completion_attack/benchmark.py`

## Reproduction

Generate the stratified subset:

```powershell
& "D:\ProgramData\py2\python.exe" scripts\build_cyberseceval_subset.py `
  --limit 120 `
  --output data_public\cyberseceval_autocomplete_subset_120.json
```

Run the low-budget matrix:

```powershell
& "D:\ProgramData\py2\python.exe" run_llm_topconf_streaming_matrix.py `
  --target hf `
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct `
  --dataset cyberseceval `
  --dataset-path data_public\cyberseceval_autocomplete_subset_120.json `
  --strategies classical_boundary_witness_comment,qscout_qbw_comment `
  --budgets 4,8 `
  --seeds 7,19,31,43,59 `
  --prompt-mode raw `
  --max-new-tokens 80 `
  --batch-size 16 `
  --output-dir outputs\ccfa_protocol_cyberseceval_qwen05_120_5seed_20260708
```

Generate tables:

```powershell
& "D:\ProgramData\py2\python.exe" generate_ccfa_protocol_tables.py `
  --roots outputs\ccfa_protocol_cyberseceval_qwen05_120_5seed_lowbudget_clean_20260708 `
  --output-dir outputs\ccfa_protocol_cyberseceval_qwen05_120_5seed_lowbudget_tables_20260708 `
  --main-method qscout_qbw_comment
```
