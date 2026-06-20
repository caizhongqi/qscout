# 最近一年方向调研与选题判断

## 结论

更推荐做 **针对 LoRA/PEFT 低秩更新的量子神经网络硬标签提取攻击**，并把抗噪 NISQ 电路作为关键实验卖点。

这个方向比“无监督 QGAN 零样本还原训练数据”更稳，因为它不需要违反量子测量的信息论限制；也比“隐蔽纠缠请求绕过经典 API”更物理合理，因为真实经典服务器不会保持攻击者查询的量子相干性。

## 四个原始创新点判断

### 方向一：无监督 QGAN 影子模型攻击

可保留：用 VQC/QGAN 作为生成式代理、减少 shadow data 依赖。

需要删除或弱化：仅凭硬标签、零样本、利用坍缩直接解调训练数据。测量坍缩不会凭空增加目标模型泄露的信息量；如果没有查询、先验或生成分布约束，这个 claim 很容易被审稿人否定。

可改写成：data-free / query-efficient quantum surrogate distillation，用公开先验或合成样本训练代理。

### 方向二：抗噪量子电路

可行，而且适合作为论文的核心实验章节。建议不要直接宣称“1% 门错误率仍可窃取真实模型参数”，而是报告在 phase flip、bit flip、amplitude damping、readout noise 下的攻击成功率曲线，并和无噪模拟器对比。

### 方向三：LoRA 参数量子注入/提取攻击

最推荐。LoRA 本身是低秩矩阵，和当前少量量子比特的表达能力匹配。攻击目标也更贴近 2025-2026 年模型安全热点：PEFT、adapter、定制模型资产保护。

建议从“LLM 全链路”先降维为“公开特征抽取器 + LoRA 线性头/adapter”的可控实验，跑通后再扩展到 transformer layer 的 LoRA。

### 方向四：双向隐蔽性纠缠攻击

不推荐作为主线。除非目标服务本身是量子黑盒并接受 coherent quantum query，否则经典 API 会在输入解析、网络传输和推理过程中破坏量子态。可以转化为“查询分布隐蔽性/低可检测性”的经典统计问题，但不要用纠缠作为核心机制。

## 当前代码原型

本项目实现了 `LoRA-QEA`：

1. 构造一个传统低秩适配分类器：`W = W_base + BA`。
2. 攻击者只查询硬标签。
3. 用物理合法的 VQC 密度矩阵模拟器生成量子特征。
4. 用 SPSA 优化量子电路参数，训练 QNN 代理。
5. 将 QNN 的响应投影回输入空间，并用 SVD 得到 rank-r LoRA 更新估计。
6. 在 phase flip 和 amplitude damping 噪声下测试攻击稳定性。

同时已经加入多数据、多模型 benchmark：

- 表格数据：synthetic low-rank classification。
- 图片数据：scikit-learn handwritten digits，8x8 灰度图。
- 时间序列数据：三类合成序列，包含正弦、方波、趋势混合模式。
- 目标模型：LoRA 低秩线性头、shallow MLP、deep MLP。

当前还没有真正的 CNN/LSTM/Transformer victim，因为本机 Python 环境没有安装
PyTorch。现有图片和时间序列目标是 MLP，而不是 CNN 或 LSTM。现在用户提供
`D:\ProgramData\py2\python.exe` 作为 PyTorch 环境后，已新增 PyTorch benchmark。
论文实验继续扩展时，应加入：

- LoRA-adapted transformer layer 或 frozen embedding + LoRA head。
- 真实图片数据集上的 CNN，例如 MNIST/CIFAR-10。
- 真实时间序列数据集上的 1D-CNN/LSTM/Transformer。

## 当前 benchmark 结果

`python run_benchmark.py` 会生成 `outputs/benchmark_results.csv`、图片样本图和时间序列样本图。快速配置使用 3 qubits、2 layers、160 hard-label queries、12 epochs。

一次本地运行结果显示：

- LoRA 低秩头：victim accuracy 约 0.709，QNN agreement 约 0.505，低秩投影模型 agreement 约 0.676。
- 表格 MLP：victim accuracy 约 0.95，QNN agreement 约 0.55。
- 图片 MLP：victim accuracy 约 0.98，QNN agreement 约 0.14。
- 时间序列 MLP：victim accuracy 约 1.00，QNN agreement 约 0.49。

这说明当前 3-qubit 快速版本更像可运行原型，不是最终强结果。图片任务 agreement 很低，后续必须提升 qubit 数、查询预算和编码方式，并加入 classical surrogate baseline。

## PyTorch 大数据 benchmark

`D:\ProgramData\py2\python.exe run_torch_benchmark.py` 使用更大的本地合成数据：

- 图片：6000 张 16x16 合成形状图，训练 4500，测试 1500。
- 时间序列：8000 条长度 64 的序列，训练 6000，测试 2000。
- 攻击查询预算：512 个 hard-label queries。
- QNN 输入：对原始输入做 StandardScaler + PCA 到 8 维，再进入 4-qubit VQC。

一次本地运行结果：

- `cnn_shape_images_16x16`：victim accuracy 1.000，QNN agreement 0.752。
- `cnn1d_long_time_series`：victim accuracy 1.000，QNN agreement 0.864。
- `lstm_long_time_series`：victim accuracy 0.891，QNN agreement 0.763。

这组结果更适合作为论文雏形中的主实验，但仍需加入真实数据集、查询预算曲线和 classical surrogate baseline。

## 论文定位草稿

题目可写成：

> QLoRA-Steal: NISQ-Aware Quantum Surrogate Extraction of Low-Rank Adapted Neural Networks

核心贡献：

- 提出第一个面向 LoRA/PEFT 低秩更新的量子代理提取攻击框架。
- 给出硬标签黑盒设定下的 VQC-to-low-rank projection 算法。
- 分析噪声通道对攻击成功率和参数恢复误差的影响。
- 讨论攻击边界：公开基座、有限查询、低秩先验、不可直接恢复训练样本。

## 开源代码调研

- `n-azimi/QShield`：2026 年 arXiv 工作，使用混合量子-经典网络提升对抗鲁棒性。可借鉴其“真实噪声模型 + 安全评估指标”的写法，但它是防御，不是提取攻击。
- `Sinestro38/qosf-qgan`：QGAN 方向的开源实现，适合借鉴生成式训练结构，但直接改成硬标签模型逆向需要较大重构。

## 近一年相关论文脉络

- [StolenLoRA: Exploring LoRA Extraction Attacks via Synthetic Data](https://arxiv.org/abs/2509.23594)：明确提出 LoRA extraction，利用公开预训练模型、合成数据和有限查询训练 substitute model。它直接支持本项目选择 LoRA/PEFT 作为攻击对象。
- [Quantum Interval Bound Propagation for Certified Training of Quantum Neural Networks](https://arxiv.org/abs/2605.00747)：QNN 安全方向的近作，说明 QNN 鲁棒性认证正在变成一个独立安全议题。本项目反过来研究 QNN 作为攻击代理。
- [A Survey of Quantum Generative Adversarial Networks](https://arxiv.org/abs/2506.18002)：总结 QGAN 结构和实现，支持 QGAN 可作为生成式代理，但也显示它更常用于数据生成，不是天然的模型逆向工具。
- [On the Generalization Limits of Quantum Generative Adversarial Networks with Pure State Generators](https://arxiv.org/abs/2508.09844)：指出纯态 QGAN 的泛化限制。这是反对“QGAN 零样本万能逆向”的重要依据。
- [The PID Controller Strikes Back](https://arxiv.org/abs/2511.14820)、[Escaping Barren Plateau](https://arxiv.org/abs/2501.13275)：都围绕 noisy VQC/barren plateau 的训练稳定性展开，支持把抗噪训练作为本项目的关键实验维度。
- [Adversarial Quantum Machine Learning](https://arxiv.org/abs/2402.00176)、[Generating Universal Adversarial Perturbations for Quantum Classifiers](https://arxiv.org/abs/2402.08648)：说明量子分类器在对抗样本、安全分析上的研究已存在，但主流对象多是“攻击 QNN”，而本项目是“用 QNN 攻击经典 LoRA 模型”，差异化更明显。

## 下一步实验

- 加入 query budget 曲线：64、128、256、512、1024。
- 加入与 classical MLP surrogate、random Fourier features 的 baseline。
- 把目标从线性 LoRA 头扩展到一层 MLP adapter。
- 增加 readout error 和 shot noise，避免只做 density-matrix exact expectation。
- 用真实 PEFT 任务的 frozen embedding 特征替换合成数据。
