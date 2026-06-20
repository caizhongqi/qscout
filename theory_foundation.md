# Theory Foundation for NQSE

## What can and cannot be claimed

NQSE is a hard-label black-box *behavioral extraction* method.  It is not a
theorem that a QNN recovers the victim's weights, LoRA matrices, or training
examples.  In fact, if a victim has `P` unknown parameters quantised to `b`
bits and answers one of `C` labels per query, a transcript of `Q` queries
contains at most `Q log2(C)` bits.  Thus exact identification requires

`Q >= ceil(P b / log2(C))`.

This elementary information bound rules out the earlier zero-query claim and
makes the low-rank setting meaningful: reducing the target description length
reduces, but does not eliminate, the required query information.

## Threat model and objective

Let the victim return only `y=f(x) in {1,...,C}`.  A public or synthetic query
distribution `D_Q` yields `S={(x_i,f(x_i))}_{i=1}^Q`.  The QNN surrogate is

`g_{theta,W}(x) = argmax_c [W phi_theta(x)]_c`,

where `phi_theta` is produced by a valid density-matrix VQC.  Its empirical
objective is cross entropy on `S`; the security metric is population agreement
`A=Pr_{x~D_test}[g(x)=f(x)]`, not parameter equality.

For a finite, quantised hypothesis class `H`, Hoeffding plus a union bound gives
with probability at least `1-delta`:

`|L_D(g)-L_S(g)| <= sqrt((log|H| + log(2/delta))/(2Q))`.

For the continuous circuit, the camera-ready paper must replace `log|H|` with a
covering-number or pseudodimension upper bound for the chosen parameter range.
This is the precise missing proof obligation; it must not be presented as
already solved.

## Noise-stability proposition

The circuit inserts independent local CPTP channels after the initial encoding
and each variational layer.  Let there be `G` noise sites and each local channel
have diamond distance at most `2p` from identity.  Telescoping the channel
composition and using `||O||_infinity <= 1` for every `Z` or `ZZ` observable
gives the sufficient bound

`|Tr[O rho_noisy]-Tr[O rho_ideal]| <= min(2, 2 G p)`.

With `M` Pauli observables measured using `S` shots, a simultaneous Hoeffding
term is

`epsilon_shot = sqrt(log(2M/delta)/(2S))`.

If the linear readout is `W`, a conservative logit perturbation is

`epsilon_logit <= ||W||_2 sqrt(M) (min(2,2Gp)+epsilon_shot)`.

Therefore a prediction with top-two logit margin larger than
`2 epsilon_logit` is certified unchanged by this model of local noise and
finite-shot estimation.  This is a stability certificate, not a fault-tolerance
claim and not a proof that the QNN is better than a classical surrogate.

`run_theory_sanity.py` implements these expressions.  The full benchmark should
save trained readouts and report certified fraction versus the observed noisy
agreement.

## The actual quantum contribution

The existing physical circuit is a **noise-aware data-reuploading VQC**:

- `RY(x)` plus nonlinear `RZ(x^2/pi)` encoding is repeated across layers;
- trainable `RX/RY/RZ` blocks supply a universal single-qubit gate family;
- selectable CNOT graphs expose an explicit entanglement/noise trade-off;
- joint `Z/ZZ` observables retain local and pairwise correlations;
- density-matrix CPTP channels preserve positivity and trace, so every
  experiment follows quantum mechanics.

This is a defensible NISQ surrogate architecture, but it is an incremental
architecture contribution by itself.  The publishable novelty must be the
combination of (i) hard-label extraction capability boundaries, (ii) a proved
and empirically calibrated noise certificate, and (iii) evidence that the
quantum design has a query/noise regime not matched by parameter-count-matched
classical baselines.  If the baseline wins everywhere, the honest paper becomes
a negative result/benchmark rather than a quantum-advantage paper.

## Literature anchors

- Cerezo et al., "Cost Function Dependent Barren Plateaus in Shallow
  Parametrized Quantum Circuits," *Nature Communications*, 2021: explains why
  cost design and locality matter for trainability.
- Wang et al., "Noise-Induced Barren Plateaus in Variational Quantum
  Algorithms," *Nature Communications*, 2021: establishes that local hardware
  noise can exponentially suppress gradients, motivating explicit noise tests.
- Caro et al., "Generalization in Quantum Machine Learning from Few Training
  Data," *Nature Communications*, 2022: gives a learning-theoretic route for
  QML generalization rather than equating training agreement with extraction.
- Abbas et al., "The Power of Quantum Neural Networks," *Nature Computational
  Science*, 2021: relates QNN expressivity to data encoding and measurement.
- "Adversarial Quantum Machine Learning," 2024 preprint: positions security
  questions for QML through explicit information and threat models.  Cite only
  after verifying the final bibliographic venue/version.

The first four are foundational rather than recent.  They should be paired in
the related-work section with 2024--2026 venue-verified QML-security papers;
do not cite an arXiv date as a top-conference acceptance.
