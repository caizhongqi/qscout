# LLM Security Baselines

This directory documents the baseline layer used by the QScout-QBW LLM
code-security experiments.  The executable implementation remains centralized in
`qlea/code_completion_attack/benchmark.py` so every method shares the same
candidate pool, detector, hard-label accounting, and output writer.

| Paper name | Strategy id | Quantum | Role |
|---|---|---:|---|
| Random Search | `fair_random_comment` | no | Uniform fair-pool sampling control. |
| Risk Prior | `fair_risk_prior_comment` | no | Lexical security-risk prior. |
| INSEC-style Fixed-Pool Search | `insec_fixed_pool_comment` | no | Fixed-pool attack-prior baseline. |
| Classical Active | `classical_active_comment` | no | Classical embedding uncertainty/diversity active baseline. |
| AOT-style Ensemble | `aot_ensemble_fixed_pool_comment` | no | Ensemble-inspired fixed-pool baseline. |
| Classical Boundary Witness | `classical_boundary_witness_comment` | no | Non-quantum boundary-witness control. |
| QScout-QBW | `qscout_qbw_comment` | yes | Objective-aligned quantum boundary witness. |

Use the registry to obtain the canonical strategy list:

```powershell
& "D:\ProgramData\py2\python.exe" - <<'PY'
from baselines.registry import csv_strategy_list
print(csv_strategy_list())
PY
```

The paper-facing runner uses the same ids:

```powershell
& "D:\ProgramData\py2\python.exe" scripts\run_llm_security_probe.py `
  --dataset securityeval `
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct `
  --budgets 4,8,16 `
  --seeds 7,19,31,43,59
```

The current committed artifact does not fabricate B=10/20/40 results.  It uses
the verified low-budget protocol B=4/8/16 for SecurityEval and LLMSecEval, and
B=4/8 for CyberSecEval.
