# Neural Collision Research Core

This directory implements the first reproducible version of the neural-collision
program.  The code is organized around claims that can be stated precisely and
checked independently.

## 1. What is a neural collision?

For a representation map

\[
F:\mathbb{R}^d\to\mathbb{R}^m,
\]

a collision is a pair `x != x'` such that

\[
F(x)=F(x').
\]

The representation map should normally be an internal feature map, not the
final classifier logits.  If a 10-class classifier maps a 64-dimensional input
directly to 10 logits, non-injectivity follows from output dimension alone and
is not an interesting structural result.  The experiments therefore analyze a
feature representation whose output width is at least the input dimension when
possible.

## 2. Exact fixed-ReLU-region certificate

Inside the interior of one ReLU activation region `R`, an affine/ReLU network
is exactly affine:

\[
F(x)=A_R x+b_R.
\]

If

\[
\operatorname{rank}(A_R)<d,
\]

then `ker(A_R)` contains a non-zero vector `v`.  Because `x` is in the interior
of `R`, there is a non-zero interval of steps `t` for which `x+t v` remains in
`R`.  Every such step satisfies

\[
F(x+t v)=F(x)+tA_Rv=F(x).
\]

`qlea/neural_collision/core.py` implements:

- exact reconstruction of `A_R` and `b_R`;
- numerical rank and nullspace;
- collision deficiency ratio
  \[
  \mathrm{CDR}=\frac{d-\operatorname{rank}(A_R)}{d};
  \]
- exact ReLU-boundary step constraints;
- a directly evaluated collision witness.

The code refuses to claim an open-region certificate if the input is
numerically on a ReLU boundary.

## 3. Structural path-rank certificate

For a sparse layered DAG with independently parameterized non-zero edges, the
generic rank of the input-output transfer matrix is the maximum number `nu` of
vertex-disjoint input-output paths.  `structure.py` computes `nu` with a
unit-capacity vertex-splitting max-flow construction.

For generic dense-layer weights we therefore test

```text
numeric rank(A_R) == structural path number nu.
```

### Scope restriction: convolution

The independent-edge generic-rank theorem does **not** transfer verbatim to a
CNN, because convolution shares parameters across many lifted edges.  The CNN
experiment therefore uses the numerical Jacobian certificate and does not
label a standard path count as a theorem for convolution.

## 4. Sparse random-network phase transition

For an equal-width sparse bipartite layer with

\[
p=\frac{\log d+c}{d},
\]

the classical random-graph perfect-matching threshold gives the critical limit

\[
\Pr(\text{perfect matching})\to \exp(-2e^{-c}).
\]

For `L` independent equal-width transitions, full path rank requires a perfect
matching in every transition, giving the product limit

\[
\exp(-2Le^{-c}).
\]

Run:

```bash
python scripts/run_neural_collision_phase_transition.py \
  --dimensions 32,64,128 \
  --c-values -4,-3,-2,-1,0,1,2,3,4 \
  --repetitions 200
```

Outputs:

- `phase_transition_sweep.csv`
- `phase_transition_collapse.png`
- `critical_residual_rate.png`
- `cdr_transition.png`

The residual plot is important: it measures perfect-matching failure that
remains after excluding isolated vertices.

## 5. Critical-regime quantum detector

Near the matching threshold, isolated vertices are the asymptotically dominant
obstruction.  An isolated-row/column witness can be searched with nested Grover
search:

1. inner search over the opposite partition checks whether a vertex has any
   incident edge, using `Theta(sqrt(d))` adjacency queries;
2. outer search looks for a vertex with no incident edge, using another
   `Theta(sqrt(d))` amplification factor.

This gives an `O(d)` adjacency-query detector for the critical isolation
obstruction.

This is **not** claimed to be a worst-case `O(d)` exact perfect-matching
algorithm.  Hall obstructions without isolated vertices remain possible at
finite size and away from the random critical regime.  The distinction is
encoded directly in `quantum.py`.

`qiskit_grover.py` contains an optional explicit Qiskit statevector circuit for
the Grover primitive.  A full coherent adjacency oracle is a separate resource
construction and is the next quantum-circuit milestone.

## 6. Trained-network validation

The cross-architecture script uses sklearn Digits, so no external download is
required:

```bash
python scripts/run_neural_collision_trained_digits.py \
  --epochs 80 \
  --instances 12
```

It trains:

- `64 -> 128 -> 128 -> 10` MLP;
- a small convolutional feature extractor with a 128-dimensional
  representation.

Then it sweeps global magnitude pruning and records:

- accuracy;
- local feature Jacobian rank;
- CDR;
- MLP path number `nu` and `rank == nu` rate;
- directly verified collision pairs.

The PyTorch adapter also works with a user-supplied feature submodule from a
larger trained network.  For smooth nonlinearities such as GELU/softmax, a
Jacobian kernel is only a first-order statement, so the adapter labels a pair as
numerically verified only after direct forward evaluation.

## 7. What the present code proves and what it does not

### Supported now

- exact local collisions for interior ReLU regions with rank deficiency;
- graph-theoretic generic path rank for independent sparse dense edges;
- reproducible matching-threshold experiments;
- an `O(d)` critical isolation-query bound under an adjacency-oracle model;
- explicit Grover statevector validation;
- trained MLP/CNN numerical collision experiments.

### Not yet supported

- a new worst-case quantum perfect-matching complexity result;
- a path-rank theorem that ignores convolutional parameter sharing;
- exact collision claims for arbitrary smooth activations from Jacobian rank
  alone;
- fault-tolerant gate counts for a data-loading/adjacency oracle without
  specifying how the sparse neural graph is encoded.

Those exclusions are deliberate.  They keep the code aligned with claims that
can survive a strong theory review.
