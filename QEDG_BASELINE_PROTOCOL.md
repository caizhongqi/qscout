# QEDG External Baseline Protocol

QEDG is an external AAAI 2025 hard-label data-free extraction baseline. It must
be executed from its official repository, not reimplemented and renamed inside
QScout. The official MNIST command uses a generator, over-confidence loss,
diversity loss, selection threshold, and a 5,000-query budget.

## Fair-comparison requirements

1. Train the same victim checkpoint and evaluate on the same held-out split.
2. Match hard-label budget, seed, query accounting, and final clone capacity.
3. Export per-seed fidelity, victim accuracy, query count, wall-clock time, and
   generated-candidate count to a CSV using QScout column names.
4. Preserve QEDG failures and incomplete runs in the appendix.
5. Do not report QEDG on UCR time series without a modality-appropriate
   generator; mark such cells `not_applicable` rather than filling zeros.

## Official source

GaozhengPei/QEDG, `main.py` and `scripts/mnist.sh`:
https://github.com/GaozhengPei/QEDG

The QScout primary comparison is now Q-CABS (quantum committee-assisted boundary
sampling): an ensemble of independently initialized physically valid QNN guides
scores candidate uncertainty, inter-guide disagreement, and geometric coverage.
This is a new in-house method, not QEDG.
