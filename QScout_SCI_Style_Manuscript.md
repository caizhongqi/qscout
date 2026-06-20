# Q-Scout: A Capability-Boundary Study of Noisy Quantum Surrogates for Hard-Label Model Extraction

**Anonymous authors**

## Abstract

Hard-label prediction APIs expose only a top-1 class, yet may still permit functional model extraction through carefully selected queries. We study whether a noisy variational quantum circuit (VQC) can serve as a query-selection surrogate for this setting. Q-Scout separates a low-dimensional quantum guide from a high-fidelity classical clone: a data-reuploading VQC ranks public candidates by surrogate uncertainty and diversity, while a capacity-matched classical clone is trained only on queried hard labels. The construction uses physically valid unitary rotations, configurable entanglement, Z/ZZ observables, and density-matrix noise channels; it does not assume coherent access to the target API or claim that measurement reveals private training data. We formalize the hard-label threat model, give information and finite-query limits that rule out unrestricted parameter recovery claims, and prescribe paired comparisons against random and classical active querying. Our currently archived pilot results establish an important boundary: stand-alone small QNN clones fail to exceed the majority baseline on MNIST, and on controlled image and sequence tasks they trail classical MLP clones by 0.341--0.441 fidelity. A finite-shot noisy replay retains 0.793--0.813 agreement relative to an analytic 0.810 reference under small simulated noise, but is not a physical-QPU result. Thus the present evidence supports an auditable experimental framework and a negative capability finding, not a general quantum advantage. The repository contains the complete study runner, failure appendix generator, and IBM replay interface required for a subsequent multi-seed evaluation.

**Keywords:** model extraction; hard-label APIs; quantum neural networks; NISQ; active querying; empirical security.

## 1. Introduction

Machine-learning services routinely expose a label-only interface. Although less informative than confidence scores, this interface can be queried adaptively and may disclose a target's decision behavior. Existing extraction work has established that query distribution, surrogate capacity, and defenses matter as much as the final classifier. Recent data-free hard-label methods, including QEDG, focus on generating informative queries without target probabilities. Meanwhile, NISQ-era quantum machine learning offers compact nonlinear feature maps but also faces finite shots, noise, and trainability constraints.

This paper asks a deliberately narrow question: can a physically realizable VQC improve *which public inputs are queried* under a fixed hard-label budget? The question differs from two stronger claims that are not justified by a classical prediction API: recovering training data from measurement collapse, and transmitting entangled queries to a target that accepts only classical inputs. Our design therefore puts all quantum computation on the attacker's side and makes the final clone classical.

The paper is written in the discipline of formal security papers: it separates interface assumptions, definitions, construction, propositions, evidence, and limitations. It is nevertheless an empirical ML-security paper rather than a new cryptographic primitive. In particular, the information bound below is a limitation theorem, not a proof that Q-Scout is secure or optimal.

**Contributions.** (i) We formalize hard-label quantum-guided extraction with a classical-only target interface. (ii) We introduce a noise-aware data-reuploading QNN as a query selector, not an unjustified high-fidelity quantum clone. (iii) We provide an evidence-gated evaluation protocol with paired classical-active controls, multi-seed figures, noise scans, defense scans, and a failure appendix. (iv) We report the available pilot evidence candidly, including negative results that invalidate stronger extraction claims.

## 2. Related Work

Model extraction from prediction APIs was established by Tramèr et al.; subsequent work studied knockoff networks, data-free extraction, and defenses based on query distributions. Pei et al. introduced QEDG for query-efficient data generation in the hard-label setting. Our planned QEDG comparison uses its released implementation, rather than calling an in-house heuristic “QEDG.”

Quantum-security work has shown that security claims depend strongly on the physical interface and noise model. Pulse-level attacks concern quantum control interfaces rather than classical ML APIs. Recent adversarial-QML evaluations similarly show that encoding choice, depth, and noise can reverse conclusions. Q-Scout is distinguished by its classical hard-label target interface and by treating quantum computation as a local, noisy query-ranking module.

## 3. Model and Definitions

Let a victim classifier be `f: X -> [C]`, exposed only through `O_f(x)=argmax_c f_c(x)`. An attacker receives a public unlabeled pool `P`, may adaptively select at most `Q` inputs, and observes a transcript `T_Q={(x_t,O_f(x_t))}_{t=1}^Q`. It has no access to confidence scores, gradients, parameters, architecture, training examples, or a coherent quantum channel to the victim.

The final clone `g_Q` is evaluated on a held-out clean distribution by functional fidelity

`F(g_Q,f)=Pr[g_Q(x)=O_f(x)]`.

We additionally report majority fidelity `F_major`, victim-correct fidelity, and `Q_tau=min{Q:F(g_Q,f)>=tau}`. A query policy improves extraction only if it improves a capacity-matched final clone under the same pool, budget, preprocessing, and seed.

**Proposition 1 (hard-label information limit).** A `Q`-query transcript from a `C`-class hard-label oracle contains at most `Q log2 C` response bits conditional on the selected queries. Consequently, it cannot by itself identify an unrestricted real-valued parameter vector.  
*Justification.* Each oracle response has at most `C` outcomes; the chain rule bounds conditional transcript entropy by the sum of response entropies. This does not preclude functional approximation using public distributional structure.

**Proposition 2 (finite-pool selection limit).** For a finite selector family `A` and clone class `H`, uniform concentration requires an uncertainty term of order `sqrt((log|A|+log|H|+log(1/delta))/Q)`. A larger circuit search space therefore requires more, not fewer, hard-label queries.  
This proposition motivates pre-registering circuit ablations and reporting failed capacity cells.

## 4. Q-Scout Construction

Q-Scout projects an input to a compact guide representation `z=P_q(x)`. For `n` qubits and `L` layers, layer `l` encodes cycling feature `(j+ln) mod d` with `RY` and `RZ` rotations, applies trainable local rotations, and applies a selected entanglement graph. The guide reads out single-qubit Z expectations and adjacent ZZ correlations. All operations are represented as unitary gates followed by standard measurements; noise is modeled by CPTP phase-flip, bit-flip, or amplitude-damping channels.

After a random warm start, the QNN fits the accumulated hard labels. For a candidate `x`, the selection score combines low top-two margin and distance from the queried set. The selected *original classical input* is sent to the target. A final MLP clone receives the same queried labels but operates on a high-fidelity classical representation. This separation makes the comparison fair: no baseline is artificially forced through the QNN bottleneck.

## 5. Evaluation Methodology

The primary matrix covers MNIST, FashionMNIST, FordA, Wafer, and ElectricDevices; CNN/MLP image victims; and 1D-CNN/LSTM time-series victims. Budgets are `{64,128,256,512,1024}` and seeds are `{7,19,31,43,59}`. The internal controls are Random, Classical-Active (MLP uncertainty plus diversity), and Q-Scout. QEDG is an external baseline that must be rerun under matched victims and budgets.

The evaluation has six pre-registered components: main fidelity-query curves; strong baseline comparisons; circuit ablations; finite-shot and noise studies; defense studies; and generalization/cost reporting. QNN-guided and classical-active clone fidelities are paired by dataset, victim, seed, budget, circuit, and defense setting. We report mean difference, normal 95% confidence interval, and win/tie/loss counts. A positive single seed is never treated as evidence of advantage.

Defense experiments apply label randomization only to attack-time responses and retain a clean victim for evaluation. The artifact records query count, selection time, QNN fit time, and total time for every row. The hardware protocol requires a backend, date, calibration snapshot, transpilation settings, shots, job ID, and mitigation setting before any physical-QPU claim is permitted.

## 6. Archived Results and Analysis

The archived results are deliberately separated from the planned main matrix. On a MNIST CNN smoke test with 32 labels, the victim accuracy is 0.990, majority fidelity is 0.150, and the stand-alone QNN achieves 0.100. It therefore fails the minimum nontriviality gate. A 256-query MNIST pilot similarly records a QNN fidelity of 0.090 against a 0.137 majority baseline, while the final MLP clone reaches 0.807. These are failures of the stand-alone QNN-clone hypothesis.

On controlled 256-query benchmarks, QNN fidelity is 0.617 for a perfect-accuracy image CNN victim compared with 1.000 for the matched MLP clone; for a long-series 1D-CNN it is 0.559 versus 1.000; for an LSTM victim it is 0.575 versus 0.917. The low-rank toy experiment also fails to recover parameters: update cosine similarity is 0.041 and relative error is 1.044. These values rule out any present claim of LoRA-parameter theft.

In a finite-shot simulation, the analytic reference agreement is 0.810. Replays yield 0.813 under phase-flip probability 0.002 with 2048 shots, 0.793 under phase-flip 0.01 with 1024 shots, and 0.795 under amplitude damping 0.01 with 1024 shots. The result supports limited circuit stability under a controlled simulator, not real-hardware extraction efficacy.

The detailed tables, exclusion decisions, and raw-file provenance are provided in `实验结果与分析_中文.md`. Existing Wafer and FordA smoke values are excluded from positive interpretation because the logs show possible majority behavior or under-trained victims.

## 7. Limitations and Responsible Interpretation

The current artifact has no completed five-seed main matrix, no matched official QEDG result, no physical-QPU job, and no evidence that Q-Scout beats classical active learning. It must therefore not be submitted as a quantum-advantage paper. Its defensible present contribution is a reproducible, physics-consistent framework for measuring when quantum query surrogates fail, what noise they tolerate in simulation, and which evidence is needed for a stronger claim.

The attack model concerns authorized research on locally trained victims or systems for which testing permission exists. It does not justify probing third-party services outside an approved security evaluation.

## 8. Reproducibility

The repository entry point is `run.py`. `python run.py --mode main --figures` executes the main study and produces evidence-bound figures only from multi-seed CSVs. `run.py --figures-only` never turns a single-seed pilot into a result figure. The failure appendix is produced by `generate_main_result_figures.py` and must accompany any reported positive table.

## References

1. Tramèr et al. Stealing Machine Learning Models via Prediction APIs. USENIX Security, 2016.
2. Orekondy et al. Knockoff Nets: Stealing Functionality of Black-Box Models. CVPR, 2019.
3. Kariyappa et al. MAZE: Data-Free Model Stealing Attack Using Zeroth-Order Gradient Estimation. NDSS, 2021.
4. Pei et al. Exploring Query Efficient Data Generation towards Data-free Model Stealing in Hard Label Setting. AAAI, 2025.
5. Xu and Szefer. Security Attacks Abusing Pulse-level Quantum Circuits. 2024.
6. Nowmi et al. Critical Evaluation of Quantum Machine Learning for Adversarial Robustness. IEEE S&P Poster, 2026.
