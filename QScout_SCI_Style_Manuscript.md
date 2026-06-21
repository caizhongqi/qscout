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

## 3. Theoretical Foundation of Q-CABS

### 3.1 Threat model and notation

Let a victim classifier be `f: X -> [C]`, exposed only through the classical oracle `O_f(x)=argmax_c f_c(x)`. An attacker receives a public unlabeled pool `P`, may adaptively select at most `Q` classical inputs, and observes a transcript `T_Q={(x_t,O_f(x_t))}_{t=1}^Q`. It has no access to confidence scores, gradients, parameters, architecture, training examples, or a coherent quantum channel to the victim. All quantum operations occur locally at the attacker.

The final clone `g_Q` is evaluated on a held-out clean distribution by functional fidelity `F(g_Q,f)=Pr[g_Q(x)=O_f(x)]`. We additionally report majority fidelity `F_major`, victim-correct fidelity, and `Q_tau=min{Q:F(g_Q,f)>=tau}`. A query policy improves extraction only if it improves a capacity-matched final clone under the same pool, budget, preprocessing, and seed.

### 3.2 Quantum state model

Let `z=P_q(x) in R^d` be a public low-dimensional representation and let `rho_0=|0><0|^(tensor n)` be an `n`-qubit initial state. Committee member `m` applies a data-reuploading circuit

`U_m(z;theta_m)=prod_(l=0)^(L-1) E_(G,l) V(theta_(m,l)) D_l(z)`,

where `D_l` contains `RY` and `RZ` encodings of cycling feature `(j+ln) mod d`, `V` contains trainable local Pauli rotations, and `E_G` is a CNOT entanglement layer. A noisy execution is

`rho_m(z)=N_p o Ad_(U_m(z;theta_m))(rho_0)`,

where `Ad_U(rho)=U rho U^dagger` and `N_p` is a local phase-flip, bit-flip, or amplitude-damping channel. The readout vector consists of Pauli observables `O_k in {Z_i,Z_i Z_(i+1)}`:

`r_(m,k)(z)=Tr[O_k rho_m(z)]`.

### 3.3 Physical validity theorem

**Theorem 1 (physical realizability).** If every `D_l`, `V`, and `E_G` is composed of standard unitary gates and `N_p` is a completely positive trace-preserving (CPTP) channel, then `rho_m(z)` is a valid density operator for every input and every committee member. Every Q-CABS readout is therefore a Born-rule expectation of a bounded observable and can be estimated on a standard gate-based NISQ device.

**Proof sketch.** The initial state is positive semidefinite with trace one. Unitary conjugation preserves both properties. CPTP maps preserve positivity and trace. Hence the final state is a density operator. Since each Pauli observable has spectrum in `{-1,+1}`, its expectation is a valid physical measurement statistic. This theorem rules out nonphysical claims such as extracting information from an unmeasured state or sending entangled queries to a classical-only API.

### 3.4 Finite-shots certificate

For `S` independent shots, write `rhat_(m,k)` for the empirical mean of observable `O_k`. Let `K` be the number of Z and ZZ observables.

**Theorem 2 (uniform measurement certificate).** For any `delta in (0,1)`, with probability at least `1-delta`,

`max_(m,k) |rhat_(m,k)-r_(m,k)| <= epsilon_S(delta)`

where

`epsilon_S(delta)=sqrt(log(2MK/delta)/(2S))`.

**Proof sketch.** Each shot of a Pauli observable is a bounded random variable in `[-1,1]`. Hoeffding's inequality bounds one empirical expectation; a union bound over `M` committee members and `K` observables gives the result. The certificate is used to distinguish true surrogate ambiguity from finite-shot fluctuation.

### 3.5 Noise sensitivity and ranking stability

Let `a_m(z)=W_m r_m(z)+b_m`, `p_m=softmax(a_m)`, and `pbar=M^(-1) sum_m p_m`. Q-CABS combines boundary uncertainty, committee disagreement, coverage, and a shot-risk penalty:

`s(z)=alpha[1-(pbar_(1)-pbar_(2))]+beta M^(-1)sum_m||p_m-pbar||_2^2+gamma C(z)-eta R_S(z)`.

Assume the score is `L_s`-Lipschitz with respect to the concatenated readout vector in the operating region. This is satisfied when readout-head norms are bounded and the top-two class ordering is locally unchanged.

**Theorem 3 (certified pairwise ranking).** Under the event in Theorem 2, the finite-shot score obeys `|shat(z)-s(z)| <= L_s epsilon_S(delta)`. For two candidates `z_a,z_b`, if

`s(z_a)-s(z_b) > 2 L_s epsilon_S(delta)`,

then their order is unchanged by finite-shot estimation with probability at least `1-delta`.

**Proof sketch.** Apply the Lipschitz condition separately to both candidates and use the triangle inequality. The strict score-gap condition leaves a margin larger than the maximum two-sided perturbation. This is the theoretical role of the shots-risk term: it penalizes candidates whose apparent uncertainty is smaller than their measurement uncertainty.

For physical noise, any Pauli observable has `|Tr[O(rho-rho')]| <= ||rho-rho'||_1`. Thus a calibrated upper bound on channel-induced trace distance can be propagated into an additional conservative term in `R_S`. This is a robustness certificate for *query ordering*, not a guarantee of victim-clone fidelity.

### 3.6 Information-theoretic and learning-theoretic limits

**Proposition 4 (hard-label information limit).** A `Q`-query transcript from a `C`-class hard-label oracle contains at most `Q log2 C` response bits conditional on selected inputs. It cannot, without additional structural assumptions, identify an unrestricted real-valued target parameter vector.

**Proposition 5 (selector complexity penalty).** For a finite selector family `A` and clone class `H`, uniform finite-query generalization incurs a term of order `sqrt((log|A|+log|H|+log(1/delta))/Q)`. Increasing circuit families or committee size may improve representation, but also increases selection complexity and cannot be presented as free information.

These propositions explain why the paper evaluates functional extraction rather than parameter recovery and why larger circuits are treated as an ablation rather than an assumed improvement.

### 3.7 What the theory does not claim

Theorems 1--3 establish physical validity and finite-shot ranking stability; Propositions 4--5 establish limits. None proves a computational quantum advantage over classical active learning. That claim remains empirical and requires positive paired multi-seed comparisons against Classical-Active and QEDG. This separation is essential: it prevents a physically valid QNN from being mistaken for a proven superior attack.

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
