# Response to External CCF-A Readiness Review

日期：2026-07-07

这份文件把外部 CCF-A readiness review 中的硬伤转成工程修复项。原则：不把弱结果包装成 CCF-A，不隐藏负结果，不声称 raw quantum advantage。

## 已修复的 artifact blocker

| Review issue | Fix |
|---|---|
| 仓库没有可复核主结果 CSV | Added `paper_artifacts/ccfa_20260707/` with main comparison, budget, AULC, seed-level, mechanism, and strict ablation CSVs. |
| `outputs/` 和 `*.csv` 被 `.gitignore` 忽略，结果无法提交 | Added `.gitignore` exceptions for `paper_artifacts/**`. |
| 依赖环境不完整 | Expanded `requirements.txt` with version ranges and added `pyproject.toml`. |
| `qlea.__init__` 顶层导入重依赖，阻断轻量 sanity | Reworked `qlea/__init__.py` to lazy-load optional experiment classes. |
| 缺少一键 artifact test | Added `scripts/verify_ccfa_artifacts.py` and `tests/test_lightweight_artifacts.py`. |
| 第二 victim / 第二 benchmark 不完整 | Added Qwen2.5-Coder-1.5B on LLMSecEval, 5 seeds, B=4/8/16, QScout vs Classical Boundary Witness. |
| 机制证据不是 5-seed | Added four 5-seed mechanism correlation tables. |
| LLM full run 太慢 | Added HF `complete_many` batching and `run_llm_topconf_streaming_matrix.py --batch-size`. |

## 当前实验证据边界

主结果现在覆盖：

```text
2 datasets x 2 victim LLMs x 5 seeds x 3 budgets
```

但必须谨慎表述：

- Qwen1.5 / LLMSecEval 的绝对 ASR 提升较小，主要贡献是 query-to-success reduction。
- strict QBW ablation 说明 raw / naked QBW 不足以支撑主 claim。
- 当前可支持的 claim 是 objective-aligned quantum query learning，而不是 end-to-end NISQ quantum advantage。

## 新增验证命令

```powershell
& "D:\ProgramData\py2\python.exe" scripts\verify_ccfa_artifacts.py
& "D:\ProgramData\py2\python.exe" run_theory_sanity.py
& "D:\ProgramData\py2\python.exe" -m unittest tests.test_lightweight_artifacts -v
```

当前本地结果：

```text
CCF-A artifact verification passed.
theory sanity passed and wrote outputs/theory_sanity.json.
unittest: 2 tests passed.
```

## 仍不能夸大的点

1. 不能写 “raw quantum boundary witness predicts utility”。
   - 机制表显示 raw `qbw_score` 多数是负相关。
   - 应写：objective-aligned acquisition converts unreliable raw quantum signals into useful query ranking evidence.

2. 不能写 “NISQ hardware validates end-to-end attack”。
   - 当前 hardware/QPU 仍是 replay/sanity，不是主结果。

3. 不能写 “所有 setting 都有 95%+ ASR”。
   - Qwen1.5 / LLMSecEval B4/B8/B16 为 82.67%, 91.47%, 93.33%。
   - 可以写 SecurityEval 和 Qwen0.5 的 B8/B16 接近或达到 95%-100%。

4. 不能把 strict ablation 当作正结果。
   - 它是边界实验，说明目标对齐层不可缺。

## 下一步必须继续补的项

1. 增加更强 SOTA-style baselines 的 official or faithful implementations。
2. 将 raw completion traces 或可重建脚本打包成 release artifact。
3. 补 defense/cost/generalization 分析，而不是只报 ASR。
4. 将论文正文重写为 query-learning / objective-alignment 主线，删除 quantum advantage 过强表述。
