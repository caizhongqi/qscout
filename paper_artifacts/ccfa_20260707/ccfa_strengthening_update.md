# QScout 2.0 CCF-A Strengthening Update

日期：2026-07-07

## 本轮新增内容

1. 增加 HF batched runner：`run_llm_topconf_streaming_matrix.py --batch-size N`。
   - 批量化按 attempt 执行，只对当前仍未成功的 active tasks 查询。
   - 不预生成后续未查询 prompt，因此不改变 hard-label query accounting。
2. 补齐第二 victim / 第二 benchmark：
   - Qwen2.5-Coder-1.5B-Instruct on LLMSecEval
   - 5 seeds: 7, 19, 31, 43, 59
   - budgets: 4, 8, 16
   - methods: Classical Boundary Witness vs QScout-QBW
3. 补齐 5-seed mechanism correlation：
   - SecurityEval / Qwen0.5
   - LLMSecEval / Qwen0.5
   - SecurityEval / Qwen1.5
   - LLMSecEval / Qwen1.5
4. 补充 strict QBW ablation：
   - 关闭 priority/objective gate 和 QSFA warmup
   - 让 density/margin/reliability 等 QBW 模块单独暴露效果

## Final Combined Main Results

| Dataset | Victim | Budget | Baseline | QScout-QBW | Gain |
|---|---|---:|---:|---:|---:|
| LLMSecEval | Qwen2.5-Coder-0.5B | 4 | 71.33% | 90.00% | +18.67 pp |
| LLMSecEval | Qwen2.5-Coder-0.5B | 8 | 83.20% | 94.93% | +11.73 pp |
| LLMSecEval | Qwen2.5-Coder-0.5B | 16 | 97.07% | 98.00% | +0.93 pp |
| LLMSecEval | Qwen2.5-Coder-1.5B | 4 | 79.20% | 82.67% | +3.47 pp |
| LLMSecEval | Qwen2.5-Coder-1.5B | 8 | 89.07% | 91.47% | +2.40 pp |
| LLMSecEval | Qwen2.5-Coder-1.5B | 16 | 92.80% | 93.33% | +0.53 pp |
| SecurityEval | Qwen2.5-Coder-0.5B | 4 | 72.40% | 91.24% | +18.84 pp |
| SecurityEval | Qwen2.5-Coder-0.5B | 8 | 83.31% | 98.02% | +14.71 pp |
| SecurityEval | Qwen2.5-Coder-0.5B | 16 | 97.19% | 100.00% | +2.81 pp |
| SecurityEval | Qwen2.5-Coder-1.5B | 4 | 73.22% | 90.08% | +16.86 pp |
| SecurityEval | Qwen2.5-Coder-1.5B | 8 | 86.45% | 99.17% | +12.73 pp |
| SecurityEval | Qwen2.5-Coder-1.5B | 16 | 98.02% | 100.00% | +1.98 pp |

## Query-Efficiency Evidence

Qwen2.5-Coder-1.5B / LLMSecEval 的绝对 ASR 提升较小，但 QScout-QBW 仍减少了查询到成功的平均步数：

| Budget | QScout Q@Success | Baseline Q@Success | Reduction |
|---:|---:|---:|---:|
| 4 | 1.42 | 1.72 | 17.34% |
| 8 | 1.80 | 2.18 | 17.15% |
| 16 | 1.99 | 2.68 | 25.57% |

这说明第二 victim 的收益更多体现在 query efficiency，而不是饱和区间的最终 ASR。

## Mechanism Evidence

关键结论：raw `qbw_score` 本身并不可靠，多数 setting 中与真实 utility 负相关；目标对齐后的 `guarded_qbw` 和 `actual_qbw_acquisition` 才稳定预测 query utility。

| Setting | Signal | Spearman | AUC | Lift@50 |
|---|---|---:|---:|---:|
| SecurityEval / Qwen0.5 | actual acquisition | 0.180 | 0.617 | 2.51 |
| SecurityEval / Qwen0.5 | raw qbw_score | -0.042 | 0.470 | 0.85 |
| LLMSecEval / Qwen0.5 | actual acquisition | 0.091 | 0.611 | 1.85 |
| LLMSecEval / Qwen0.5 | raw qbw_score | -0.125 | 0.388 | 0.21 |
| SecurityEval / Qwen1.5 | actual acquisition | 0.180 | 0.612 | 1.44 |
| SecurityEval / Qwen1.5 | raw qbw_score | -0.028 | 0.477 | 1.04 |
| LLMSecEval / Qwen1.5 | actual acquisition | 0.290 | 0.730 | 1.49 |
| LLMSecEval / Qwen1.5 | raw qbw_score | -0.119 | 0.414 | 1.23 |

论文写法应强调：QScout-QBW 的贡献不是“裸量子分数”，而是 objective-aligned quantum boundary acquisition。

## Strict Ablation Finding

strict ablation 关闭了 objective anchors、priority gate 和 QSFA warmup。结果显示 strict QBW alone 明显弱于完整 QScout-QBW 和 Classical Boundary Witness：

| Budget | Classical BW | Strict Full QBW | Strict Random Quantum |
|---:|---:|---:|---:|
| 4 | 72.40% | 68.76% | 66.94% |
| 8 | 83.31% | 72.07% | 72.07% |
| 16 | 97.19% | 77.69% | 74.88% |

这不是主结果失败，而是一个重要边界结论：量子态边界信号不能脱离目标对齐层单独使用。完整 QScout-QBW 的收益来自：

```text
 fair candidate pool
 objective-aligned hard-label feedback
 guarded quantum boundary witness
 reliability penalty
 low-budget rescue/priority constraints
```

## Updated Artifact Paths

| Artifact | Path |
|---|---|
| Final combined main table | `outputs/ccfa_protocol_combined_qwen05_qwen15_5seed_final_tables_20260707` |
| Qwen1.5 LLMSecEval batched run | `outputs/ccfa_protocol_llmseceval_qwen15_2method_5seed_batched_20260707` |
| Qwen1.5 LLMSecEval table | `outputs/ccfa_protocol_llmseceval_qwen15_2method_5seed_batched_tables_20260707` |
| SecurityEval/Qwen0.5 mechanism | `outputs/ccfa_mechanism_5seed_securityeval_qwen05_20260707` |
| LLMSecEval/Qwen0.5 mechanism | `outputs/ccfa_mechanism_5seed_llmseceval_qwen05_20260707` |
| SecurityEval/Qwen1.5 mechanism | `outputs/ccfa_mechanism_5seed_securityeval_qwen15_20260707` |
| LLMSecEval/Qwen1.5 mechanism | `outputs/ccfa_mechanism_5seed_llmseceval_qwen15_20260707` |
| Strict QBW ablation | `outputs/ccfa_qbw_strict_ablation_securityeval_qwen05_5seed_tables_20260707` |

## 当前判断

主结果矩阵现在已经更接近 CCF-A：两个公开 benchmark、两个 open-source victim LLM、5 seeds、低预算 query-efficiency 曲线、机制相关性和负/边界消融均已补上。

仍需谨慎表述的是：Qwen1.5 / LLMSecEval 已接近饱和，绝对 ASR 提升较小；该 setting 应作为 robustness/query-efficiency evidence，而不是最大效果展示。
