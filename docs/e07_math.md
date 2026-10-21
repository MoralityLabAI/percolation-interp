# Percolation and polysemanticity: what the saved VPD can test

## The mathematical object

The saved decompositions express a weight matrix as rank-one subcomponents,
plus a residual delta component. In the implementation, a linear subcomponent
maps an input x to `(x @ V[:, c]) * U[c, :]`. This follows the
[VPD parameterization](https://arxiv.org/html/2610.20748v1#S2.SS1).
The delta component is not included in the geometry study; consequently these
graphs are not a representation of every parameter in the full model.

For an MLP neuron coordinate j and subcomponent c, let a_cj be its write
coefficient (U for gate/up) or read coefficient (V transposed for down).
Define the normalized energy

    q_cj = a_cj^2 / sum_l a_cl^2.

This is invariant to the per-component scalar ambiguity in a rank-one
factorization. It ignores sign and amplitude. Two descriptive quantities are

    coordinate spread of component c = 1 / sum_j q_cj^2,
    effective components at coordinate j = (sum_c q_cj)^2 / sum_c q_cj^2.

The second quantity is a participation ratio for **unit-energy directions**.
It is not an estimate of how many semantic features activate at a neuron.
Component norm, firing rate, activation covariance and sign cancellation are
not part of this statistic.

Connect c to its k largest positive-energy neuron coordinates. This gives a
bipartite graph: components on one side, MLP neuron coordinates on the other.
A cluster groups components that share coordinates, directly or indirectly.
Measure its mass by the number of component vertices, not the combined count
of components and coordinates. A large cluster is a structural mixing region;
it does not establish a causal circuit or a family of related meanings.

The artifact contains nominally 768 subcomponents per MLP matrix and 3,072
neuron coordinates. This is **not overcomplete by that count in an individual
role**. It does not demonstrate that there are fewer semantic features than
neurons: one subcomponent might contain several concepts, and concepts need
not correspond one-to-one to any chosen factorization.

## Where the percolation threshold comes from

For a configuration ensemble, let D_C and D_N be component and neuron degrees.
Following a random edge encounters size-biased degree distributions. The
expected numbers of *other* edges available at the two endpoint types are

    kappa_C = E[D_C(D_C-1)] / E[D_C],
    kappa_N = E[D_N(D_N-1)] / E[D_N].

A two-step exploration returns to the component side with expected branching
factor B = kappa_C * kappa_N. With independent bond retention probability p,
both edges in a two-step continuation must survive, so

    B(p) = p^2 * B,        p_c = 1 / sqrt(B).

If B=0 there is no generic branching transition; if p_c>1 it is outside the
accessible retention range. This is the sparse, locally tree-like,
uncorrelated configuration-ensemble argument, extending the bipartite
generating-function framework of
[Newman, Strogatz and Watts](https://arxiv.org/html/cond-mat/0007235v2#S4).
It is not an exact threshold for a particular finite learned graph.

The full prediction uses the ordinary degree generating functions G_C0,G_N0
and their size-biased excess functions G_C1,G_N1. The smallest fixed point is

    x = 1-p + p G_N1(y),
    y = 1-p + p G_C1(x),
    S_C = 1-G_C0(x).

Here S_C is the asymptotic fraction of component vertices in the giant cluster.
The implementation measures finite clusters independently of this prediction.
Its susceptibility is the component-mass-weighted finite-cluster size after
excluding one largest cluster. A 50%-cluster crossover is descriptive; it is
not the critical point p_c.

For a uniform fixed-k support null with C components and N coordinates,
D_C=k while D_N is Binomial(C,k/N). Thus

    E[B] = (k-1)(C-1)k/N.

At C=768, N=3,072, k=2 is subcritical in this approximation and k=3 is
supercritical. At k=8, the predicted bond threshold is approximately 0.267.
This is a null-model prediction fixed before inspecting learned supports.
It is not an empirical estimate from the VPD.

## Why the controls matter

The uniform-support null preserves the number of contacts per component and
tests coordinate concentration. The configuration null preserves both degree
sequences and tests structure beyond this concentration. It permits parallel
edges, which are occupied independently; their frequency is explicitly
reported. This finite multigraph difference limits comparisons with a simple
learned support graph.

An orthogonal coordinate mix preserves direction inner products but can change
top-coordinate supports and participation ratios. It isolates basis dependence
of the diagnostic. MLP neuron coordinates are a privileged basis because of
their nonlinearities; the coordinate mix is **not** claimed to preserve the
network function.

A useful counterexample is a star: arbitrarily many components can share one
coordinate even when every component has degree one and B=0. Its giant finite
cluster violates the non-macroscopic-hub assumptions of the generic ensemble
criterion. Direct graph measurement and the degree-preserving null are therefore
necessary, rather than interpreting B alone as a semantic threshold.

## What would establish a stronger polysemanticity claim

[Toy Models of Superposition](https://arxiv.org/abs/2209.10652) connects
polysemantic neurons to sparse features sharing a representation. That
motivates this graph, but does not supply semantic labels for these factors.
The VPD paper also discusses rank-one subcomponents that participate in more
than one computation, so "subcomponent = semantic feature" cannot be assumed.

For a stronger test, take a fixed set of high-contact neuron coordinates from
this diagnostic, collect token contexts and CI-weighted component contributions,
and check whether distinct interpretable context families share those neurons.
Control for firing rate, activation magnitude and position-0 effects. Then
intervene on the proposed component/context pairs, measuring held-out output
changes. Distinct labels without contribution measurements would be weak;
static overlap without labels is only structural evidence.

The current e07 experiment is fixed in `e07_design.md`. Its measured artifact
results are pending while the system-memory admission guard holds its CPU
job in the VRAMorama queue. No semantic or percolation finding is claimed from
the mathematical setup alone.
