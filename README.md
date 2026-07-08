# QScout

QScout is an objective-aligned quantum query learning prototype for hard-label
model extraction.  The current CCF-A-oriented branch focuses on code LLM
security benchmarks: QScout-QBW uses guarded quantum boundary-witness signals to
rank which hard-label queries are most informative.

The repository also contains older image-classification extraction experiments;
those are kept as legacy pilots and should not be treated as the current main
claim.

## Setup

```powershell
& "D:\ProgramData\py2\python.exe" -m pip install -r requirements.txt
```

QPU replay is optional and not required for the CCF-A table artifact check:

```powershell
& "D:\ProgramData\py2\python.exe" -m pip install -r requirements-qpu.txt
```

For a minimal artifact check that does not require torch, transformers, or QPU
libraries:

```powershell
& "D:\ProgramData\py2\python.exe" scripts\verify_ccfa_artifacts.py
& "D:\ProgramData\py2\python.exe" -m unittest tests.test_lightweight_artifacts -v
```

## Current CCF-A Evidence Package

The current table-level evidence is committed under:

```text
paper_artifacts/ccfa_20260707/
```

It contains:

- two public security benchmarks: `SecurityEval`, `LLMSecEval`;
- two open-source victim LLMs: `Qwen2.5-Coder-0.5B-Instruct`,
  `Qwen2.5-Coder-1.5B-Instruct`;
- five seeds: `7,19,31,43,59`;
- low-query budgets: `4,8,16`;
- strongest-baseline comparison, AULC summaries, mechanism correlation, and
  strict QBW ablation tables.

The combined main table is:

```text
paper_artifacts/ccfa_20260707/main_strongest_baseline_comparison.csv
```

Regenerate the compact final tables from local raw outputs:

```powershell
& "D:\ProgramData\py2\python.exe" generate_ccfa_protocol_tables.py `
  --roots outputs/ccfa_protocol_securityeval_qwen05_seed7_full_gatefix_20260706,outputs/ccfa_protocol_llmseceval_qwen05_seed7_full_gatefix_20260706,outputs/ccfa_protocol_securityeval_qwen15_5seed_full_gatefix_20260707,outputs/ccfa_protocol_llmseceval_qwen15_2method_5seed_batched_20260707 `
  --output-dir outputs/ccfa_protocol_combined_qwen05_qwen15_5seed_final_tables_20260707 `
  --main-method qscout_qbw_comment
```

Run the fast batched second-victim LLMSecEval comparison:

```powershell
& "D:\ProgramData\py2\python.exe" run_llm_topconf_streaming_matrix.py `
  --target hf `
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct `
  --dataset llmseceval `
  --strategies classical_boundary_witness_comment,qscout_qbw_comment `
  --budgets 4,8,16 `
  --seeds 7,19,31,43,59 `
  --prompt-mode instruction `
  --max-new-tokens 80 `
  --batch-size 8 `
  --output-dir outputs/ccfa_protocol_llmseceval_qwen15_2method_5seed_batched_20260707
```

Important claim boundary: the strict ablation shows that naked quantum
boundary-witness scores are not sufficient.  The supported claim is
objective-aligned quantum query learning, not raw quantum uncertainty.

## Run

```powershell
& "D:\ProgramData\py2\python.exe" run.py --mode smoke
```

Use `run.py --help` for available study and figure-generation commands.

## Legacy Image Candidate

The current MNIST candidate uses a four-qubit VQC in two roles: Jensen-Shannon
committee disagreement for hard-label query selection, and final VQC Born
probabilities as quantum side features for the clone.  The quick 128-query
reproduction is:

```powershell
& "D:\ProgramData\py2\python.exe" run_active_hardlabel_benchmark.py `
  --datasets MNIST --victims cnn --budgets 128 --seeds 7,19,31 `
  --strategies qedg_js_qfeature_active --victim-epochs 3 --final-epochs 1 `
  --warm-epochs 5 --initial-queries 32 --batch-size 32 --candidates 256 `
  --diversity-weight 0.25 --committee-size 2 `
  --committee-disagreement-weight 0.20 --class-balance-weight 0.15 `
  --features 16 --qubits 4 --layers 3 --eval-samples 300 `
  --clone-hidden 64 --quantum-feature-mode probs
```

The current main 256-query result uses the same configuration with
`--budgets 256` and writes to `outputs/qedg_js_qfeature_warm5_q4_multiseed_256.csv`.
It reaches 86.67% ± 1.33% agreement on the 3-seed MNIST/CNN pilot.

## 95% frontier branch

The current 95%-oriented frontier is not the Born-feature branch.  It uses the
JS quantum committee with stronger query-free augmentation:

```powershell
& "D:\ProgramData\py2\python.exe" run_active_hardlabel_benchmark.py `
  --datasets MNIST --victims cnn --budgets 512 --seeds 7 `
  --strategies qedg_js_active classical_active random `
  --victim-epochs 3 --final-epochs 1 --warm-epochs 3 `
  --initial-queries 64 --batch-size 128 --candidates 256 `
  --diversity-weight 0.25 --committee-size 2 `
  --committee-disagreement-weight 0.20 --class-balance-weight 0.15 `
  --features 16 --qubits 4 --layers 3 --eval-samples 300 `
  --clone-hidden 64 --augmentation-level strong
```

This reaches 94.00% on MNIST/CNN in the current single-seed frontier test.
Wafer reaches 98.67% at 512 queries, but must be reported with its majority
baseline because the task is highly imbalanced.

For matched 256-query controls:

```powershell
& "D:\ProgramData\py2\python.exe" run_active_hardlabel_benchmark.py `
  --datasets MNIST --victims cnn --budgets 256 --seeds 7,19,31 `
  --strategies qedg_js_active classical_active classical_latent_active random `
  --victim-epochs 3 --final-epochs 1 --warm-epochs 5 `
  --initial-queries 32 --batch-size 32 --candidates 256 `
  --diversity-weight 0.25 --committee-size 2 `
  --committee-disagreement-weight 0.20 --class-balance-weight 0.15 `
  --features 16 --qubits 4 --layers 3 --eval-samples 300 `
  --clone-hidden 64
```

See `MODEL_ITERATION_RESULTS.md` for accepted and rejected ablations.

## Security-paper metrics

`run_active_hardlabel_benchmark.py` now reports the metrics commonly expected
in AI Security / USENIX Security / NDSS style model-extraction papers:

- `fidelity`: agreement between victim and surrogate predictions.
- `success_at_threshold`: whether fidelity reaches `--success-threshold`
  (default `0.90`).
- `query_efficiency`: `fidelity / query_count`.
- `fidelity_per_1k_queries`: scaled query efficiency for readability.
- `*_qtau.csv`: first query budget at which each method reaches the requested
  fidelity threshold.

## Transformer victims

Classification attacks can now target compact Transformer victims:

```powershell
& "D:\ProgramData\py2\python.exe" run_active_hardlabel_benchmark.py `
  --datasets MNIST --victims transformer --budgets 64,128,256 `
  --seeds 7,19,31 --strategies random classical_active qedg_js_active `
  --victim-epochs 3 --final-epochs 1 --warm-epochs 3 `
  --initial-queries 32 --batch-size 32 --candidates 256 `
  --features 16 --qubits 4 --layers 3 --eval-samples 300 `
  --success-threshold 0.90
```

The current strongest Transformer branch is
`qedg_js_consensus_active`, a quantum-classical consensus query policy.  It
combines VQC committee JS/QMMI with a classical surrogate uncertainty gate while
keeping the same hard-label query budget.

For YOLO or other object detectors, use the separate protocol in
`YOLO_DETECTION_EXTRACTION_PLAN.md`.  The detector branch should fix YOLO26 as
the newest Ultralytics victim family, add YOLO11 as a stable baseline, and
report box/class agreement plus mAP-style fidelity rather than classification
fidelity.

## Layout

- `qlea/`: quantum simulator, QNNs, query selection, datasets, and victims.
- `run.py`: unified project entry point.
- `run_active_hardlabel_benchmark.py`: benchmark implementation.
- `run_study_matrix.py`: study-matrix runner.
- `generate_*.py`: figure generation utilities.

## License

MIT.
