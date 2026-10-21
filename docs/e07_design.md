# e07: component/neuron sharing and percolation (exploratory)

Written 2026-10-22 before computing e07 artifact statistics. This is a new
exploration, not a revision of any e05/e06 hypothesis or quality criterion.

Use the two existing batch-32 Qwen3-0.6B VPD final checkpoints. In each of
layers 0 and 1, examine three maps into the 3,072 MLP neuron coordinates:
the gate projection's U rows, the up projection's U rows, and the down
projection's V columns (transposed to rows). Analyze each role separately.
All nonzero subcomponents count: this is static parameter geometry, not a
claim about which units fire. In particular, no firing-rate data are assumed
for seed 1 while its analysis is pending.

Normalize each row's squared coordinates to unit energy. This removes the
rank-one factorization's per-component scalar gauge. Measure participation
ratios for components and coordinates, and energy retained in the largest k
coordinates. Use k in {1,2,3,4,6,8,12,16}; do not select k based on results.
The bipartite graph connects a component to its top k positive-energy neuron
coordinates. Never introduce edges at zero entries.

Primary bond-percolation graph: k=8. Retain each edge independently with
probability p, on p=0,.025,...,1. Use 32 nested Monte Carlo draws, fixed RNG
seed 71022. Measure largest-cluster component fraction and susceptibility
after excluding one largest cluster. The denominator counts components,
not coordinate vertices. Compare with (a) independent uniform row supports
(same component degrees), (b) configuration rewiring of coordinate stubs
(both degree sequences preserved, parallel edges permitted and measured),
and (c) one seeded orthogonal coordinate mix (three signed/permuted
block-Hadamard rounds). The mix preserves row inner products; it is not a
function-preserving transformation of the nonlinear MLP. The k sweep has
eight draws per null; report descriptive ranges, not discovery p-values.

For the sparse locally tree-like configuration ensemble, derive the
two-type branching product B=kappa_component*kappa_neuron and bond threshold
p_c=1/sqrt(B), plus the probability-generating-function giant fraction.
Compare predictions with the finite learned and rewired graphs; do not call
the degree threshold exact for the learned graph. Validate the simulation
against independent sparse connected-components calculations and known
regular bipartite fixed points. Include a large random-support calibration.

Sources: [VPD paper](https://arxiv.org/html/2610.20748v1),
[Toy Models of Superposition](https://arxiv.org/abs/2209.10652),
[Newman, Strogatz and Watts](https://arxiv.org/html/cond-mat/0007235v2).
Definitions and inferences about this artifact are ours. This study cannot
establish semantic polysemanticity: VPD subcomponents need not be individual
semantic concepts, and static direction overlap is not coactivation or
causal interference. Signed contributions, activation covariance and
context-level labels would be needed for those stronger interpretations.

Runs on CPU from memory-mapped checkpoints while the separate e05b/e06 GPU
pipeline continues. Uses no new training, rentals or model downloads.
