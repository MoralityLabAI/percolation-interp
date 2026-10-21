# e06 (DRAFT): null models for the sieve's z-curve

> **DRAFT, written 2026-10-21. It is not pre-registered.** It needs the user's
> decisions on the open choices at the end. It becomes binding only when it is
> committed as final, before any held-out graph has been analysed with the e06
> diagnostics.
>
> It was written after the exploratory analysis in `e06_exploratory.md`, which used
> e03 seed 0 and e04 seeds 0-1 as discovery data. Those graphs cannot serve as
> confirmatory data for anything below.

## Why e06

e03-e05 sieve attribution graphs against within-layer weight shuffles. The z-curve
always has two regimes:
- a positive excess at small kept degree c;
- a deep deficit at large c ("hub concentration").

The exploratory analysis suggests a different reading from "connected circuits":

1. **The small-c excess is node-level alignment.** The nodes that have a top-ranked
   input also have a top-ranked output (the Cov term below). In all four discovery
   graphs, Cov carried 64% to more than 100% of the excess.
2. **The large-c deficit is coverage.** The top edges pile onto a few hubs, so fewer
   nodes have any kept in-edge or out-edge.
3. **Path structure beyond degrees was not detectable.**
   - With residual skip edges and a single output, a node is on the backbone exactly
     when it has a kept in-edge and a kept out-edge (share ≥ 0.995 at every tested c).
     So the joint-degree null had zero spread in 11 of 12 cells.
   - In the one informative cell, real and null agreed (z +0.9).
4. **Firing frequency is a confound.** It explains the SAE graph's excess and the VPD
   graph's in/out strength correlation.

e06 tests these readings on fresh graphs. It uses fixed definitions, fixed c
values and pass/fail rules.

## Definitions

- **Graph, ranking, c grid, interior nodes:** as in e05, via `percinterp.sieve`
  (`degree_entry`, ranks within the destination layer). Interior nodes are those
  that are neither sources nor the sink.
- **Decomposition** (`diagnostics.decompose`). A_v = 1[F(v) ≤ c], D_v = 1[B(v) ≤ c].
  - For each layer: b_l = r^F_l r^B_l + Cov_l(A, D), with population covariance over
    the layer's interior nodes.
  - Totals are weighted by node count. "Coverage" = Σ w_l r^F_l r^B_l; "alignment" =
    Σ w_l Cov_l.
  - Against a null, Δb = Δcoverage + Δalignment exactly.
- **Degree-determined share S(c)** (`diagnostics.degree_determined_share`): the share of
  interior nodes whose backbone membership equals "≥ 1 kept in-edge and ≥ 1 kept
  out-edge".
- **Strength correlation** (`diagnostics.strength_correlation`): Spearman ρ between
  in-strength and out-strength. Ranks are taken within each layer and pooled over
  interior nodes. The partial version residualises both ranks on the node's
  firing-rate rank.

## Nulls

| Null | What it keeps | What it randomises | Draws |
|---|---|---|---|
| **N-JD, joint-degree** (primary for the wiring question) | At fixed c: every node's kept in- and out-degree within each (source layer, destination layer) block. Skeleton edges only. | Who connects to whom. Double-edge swaps, 30 rounds. | 1,000 per c |
| **N-DC, in/out decoupling** (primary for the alignment question) | Every edge weight, every rank, each layer's in- and out-strength sequences | Which node's inputs go with which node's outputs: out-edge sets re-dealt within the layer | 1,000 |
| N-WL, within-layer (the e03-e05 null) | Each destination layer's weight distribution, the skeleton | Weights within the destination layer | 1,000 |
| N-OS / N-IS, out- / in-strength | Each source's (target's) weight multiset | Its weights among its own edges | 1,000 each |
| N-LP, layer-pair | Each (source layer, destination layer) weight distribution | Weights within the pair | 1,000 (skipped if the same as N-WL) |
| N-AB, activity-binned | Weights within (destination layer, source rate quintile, target rate quintile) | Weights within those cells | 1,000 |

Firing rates are the alive-threshold rates from the stats rows (`stats.pt`, as in
e04/e05). The output node's rate is set to 1.

## Choice of c

Fixed now from the discovery data:
- **c_s = 0.1** (small c). Discovery c* values were 0.020-0.204, geometric mean 0.077.
- **c_L = 4** (large c). In all four discovery graphs z ≤ −95 at c = 4.
- **c = 1** is reported, and used by P4 alongside c_s.

Grid-wide statements use the sieve's max-statistic: the graph's peak z against each
null draw's own peak z measured against the remaining draws. That corrects for the
48-point grid in the same way as e03-e05's Q1.

## Data

**Held-out graphs (choose; see the open choices):**
- **e05b seeds 0 and 1:** Qwen3-0.6B layers 0-1, VPD at batch 32. These are fresh
  only if this design is committed as final before either graph is analysed with
  any e06 diagnostic.
  - e05b's own pipeline will compute the within-layer and in-strength sieve curves
    (e05's Q1). Those are e05 quantities, and they will probably be seen first. The
    predictions below avoid depending on them, except P3, which uses the
    within-layer null at c_L.
- **Optionally, e03 seeds 1 and 2** (toy model). Their graphs exist. Their
  within-layer and in-strength sieve curves were seen in `e03_replication.md`, but
  no e06 diagnostic has been run on them. That makes them partly seen: they can
  confirm P1, P2, P4 and P5, but not P3.

Cost: CPU only. A Qwen-slice graph (≈ 10^5 edges) takes seconds per 100 draws; e03
VPD (3.6M edges) about 2 minutes per 100. With 1,000 draws per null, the whole study
is a few CPU hours, mostly on the e03 VPD graphs.

## Predictions

Each prediction is scored separately per held-out graph. A prediction **holds** if
it passes on every held-out graph. The discovery values (from `e06_exploratory.md`
and the fixed-c rerun) are given for reference only.

**P1: the small-c excess is alignment.**
- Statistic: Δb and Δalignment at c_s against N-WL.
- Pass: Δb > 0, Δalignment > 0, and Δalignment ≥ 0.5 × Δb.
- Discovery: e03 VPD 0.036 of 0.056 (64%); e03 SAE 0.013 of 0.006; e04 0.010 of
  0.008, and 0.005 of 0.002.
- This tests composition, not significance. Against N-WL, the excess at c_s was
  significant only in VPD (z 18.9; 0.6-2.5 elsewhere).

**P2: alignment beyond chance pairing.**
- Statistic: backbone fraction at c_s against 1,000 N-DC draws.
- Pass: one-sided p ≤ 0.01, with the +1 correction.
- Secondary, reported: the max-statistic version over the whole grid.
- Discovery z at c = 0.1: 19.5, 9.6, 4.1, 3.2.

**P3: the large-c deficit is coverage.**
- Statistic: Δb, Δcoverage and Δalignment at c_L against N-WL.
- Pass: z ≤ −3, Δcoverage ≤ Δb (coverage accounts for at least the whole deficit),
  and Δalignment ≥ −0.1 × |Δb|.
- Discovery: Δb −0.29, −0.62, −0.52, −0.55; Δalignment 0, +0.10, 0, 0.
- Partly definitional: once all interior nodes are reached, Cov is 0. Reported
  alongside, not tested: which side loses coverage (r^F against r^B), and N-OS / N-IS
  z at c_L.

**P4: no wiring beyond kept degrees** (the joint-degree null).
- Cells: c_s and c = 1. S(c) is reported per cell.
- A cell is **not testable** if the N-JD backbone fractions have zero spread.
- In a testable cell, the "wiring excess" test is real > null, one-sided, with
  1,000 rewirings and Bonferroni over the two cells: p ≤ 0.005.
- **P4 passes** if no testable cell shows a wiring excess. It is reported as "not
  testable" if neither cell is testable.
- Discovery: 11 of 12 cells untestable. In the testable cell, z +0.9, p = 0.25.

**P5: the in/out strength alignment is mostly firing frequency.**
- Pass: pooled ρ(in, out) > 0 with p ≤ 0.01, against 1,000 within-layer permutations
  of out-strength.
- **And** the rate-partialled ρ is less than half of ρ.
- Discovery: VPD 0.65 → −0.02; Qwen 0.37 → 0.13 and 0.26 → 0.08; SAE 0.85 → 0.43
  (would fail, at 0.51).

**Reported, not tested:**
- every null's z-curve and peak (N-WL, N-OS, N-IS, N-LP, N-AB, N-DC);
- the decomposition curves;
- kept-degree coverage against the uniform-placement formula N[1 − (1 − 1/N)^m];
- backbone anatomy at c* and at c = 1: nodes per module, edge layer-spans, path
  counts by hop length;
- whether the small-c excess survives N-AB (discovery: VPD yes, z 10.0; SAE no,
  0.97; Qwen borderline, p 0.06 and 0.12).

## Reading the result

- **P1, P2 and P3 hold; P4 passes or is untestable:** the sieve's signal is
  node-level alignment plus coverage loss from hubs. Future sieve tests should use
  N-DC, not N-WL, as the control for "structure". They should stop describing the
  backbone as a circuit of paths unless a non-degenerate N-JD shows wiring.
- **P2 fails:** the small-c excess is no larger than chance pairing of a layer's
  inputs and outputs. The sieve's positive signal does not transfer.
- **P4 finds wiring in a testable cell:** there is path structure beyond degrees.
  This is the "circuit" reading, and it would need its own follow-up.
- **P5 fails** (alignment survives rate control): alignment is more than
  frequency. N-AB then becomes the stricter control for future work.

## Open choices for the user

1. **Primary null.** The brief named the joint-degree null as primary. On e04-like
   topologies it is degenerate, and e05b's slice is e04-like. This draft makes it
   primary for the wiring question (P4, with a "not testable" outcome), and makes
   in/out decoupling primary for the alignment question (P2). The alternative is
   N-JD only, accepting that it will probably return "not testable" on e05b.
2. **c selection.**
   - (a) Fixed c_s = 0.1 and c_L = 4, as drafted.
   - (b) Each held-out graph's own within-layer c*, determined by the sieve's rule.
   - (c) Max-statistic over the grid everywhere.
3. **Held-out data.**
   - e05b only: then this design must be committed before e05b's graphs exist, and
     e05b's local analysis starts unattended when the decompositions arrive tonight.
   - e05b plus e03 seeds 1-2.
   - Or decide that seeing e05b's within-layer sieve curve first is acceptable (it
     touches P3).
4. **Gate dependence.** Should e05b graphs count for e06 even if their decomposition
   fails e05's quality gate? e06 is about the graphs' null structure, not about
   transfer.
5. **Thresholds and draws.** 1,000 draws and p ≤ 0.01 as drafted (0.005 per cell
   for P4), or stricter.
6. **Scope.** Keep P5 (frequency) tested, or report it only. Promote N-AB to a test.
7. **Swap sampler.** The parallel double-edge swap is well mixed in practice, but it
   is not an exact uniform sampler. Accept it, or require sequential Maslov-Sneppen
   swaps with a mixing check (slower on VPD-size graphs).

## Order of work (once approved)

1. Commit the final version of this design.
2. When the held-out graphs exist, run `experiments/e06_explore.py` with 1,000
   draws on them only. A held-out mode with the fixed c values still has to be added.
3. Score P1-P5 and write `e06_results.md`.
