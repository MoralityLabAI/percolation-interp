# e04 results: the percolation sieve on a Qwen3-0.6B slice

Design and predictions: `e04_design.md`, committed before the decomposition was
trained. Its revisions (the slice moved to layers 2-3; the held-out rows split
three ways) and the gate addendum were all made before any Q1-Q3 quantity
existed. Run 2026-10-18.

## Verdict

**Not transfer, in either seed.**

- The decomposition passes its quality gate in both seeds.
- Q1, the connectivity test, fails in both.
- Q2 and Q3 pass in both. But in each seed one layer-2 attention pair carries
  nearly all of the backbone's effect, and in seed 1 both passes come down to
  0.002 nats.

| Test (rule) | Seed 0, `p-1ae4ebad` | Seed 1, `p-542cdc24` (pre-registered replication) |
|---|---|---|
| Quality gate (at most 10% of the gap unrecovered) | 7.7%: pass | 8.3%: pass |
| Q1, within-layer shuffles (beat all 100) | peak z 4.2, shuffles at most 3.7: pass | peak z 3.0, 6 shuffles at or above it, max 4.8 (p = 0.07): **fail** |
| Q1, in-strength shuffles (beat all 100) | peak z 4.5, 1 shuffle above it at 7.3 (p = 0.02): **fail** | peak z 3.2, 3 shuffles at or above it, max 4.5 (p = 0.04): **fail** |
| Q2, KL below all 100 matched random sets | 6.32 vs best 6.41: pass | 6.433 vs best 6.435: pass |
| Q3, KL below the same-size top mean-CI set | 6.32 vs 6.40: pass | 6.433 vs 6.435: pass |

- **Q1 fails in both seeds.** Seed 0 missed one control by a single outlying
  shuffle. Seed 1 misses both controls clearly, so seed 0's within-layer pass
  did not replicate.
- **The design named "transfer" (all pass) and "partial" (Q1 passes, Q2 or Q3
  fails).** Neither seed is either: the connectivity test fails while the
  causal tests pass.
- **The causal passes rest on one attention pair** (exploratory section and
  replication below).
  - Each seed's backbone contains a value and an output direction of layer 2's
    attention that together close about 4% of the gap. Either one alone closes
    nothing.
  - In seed 1 the backbone is that pair plus components adding 0.04 points. The
    best random set and the top mean-CI set contain the same pair.
- **The structure the sieve finds is small and weak.**
  - Seed 0: 12 components, 2% of the 599 alive. Seed 1: 6 components, of 873.
  - The peak z is 3.0-4.5, against 28.7 on e03's toy model.

The sections from "What ran" to "What the data show" describe seed 0. Seed 1 is
under "Replication".

## What ran

- **Decomposition.** Run `p-1ae4ebad`: VPD of layers 2 and 3 of Qwen3-0.6B,
  all seven projections each, 7,680 components, seed 0, 20,000 steps.
  - It was the third attempt. The first was killed by a Windows Update restart
    at step 3,689. The second hung at step 19,795 after a charger power flap
    wedged its CUDA context. Neither left a final checkpoint.
  - The third ran under a watchdog driver (`experiments/e04_pipeline.py`) and
    finished in 3 h 27 min without a restart.
- **Quality** (param-decomp's evaluation at step 20,000, `results/e04/gate.json`):

  | Masking | KL to the target (nats) | Share of the zero-ablation CE gap unrecovered |
  |---|---|---|
  | CI masks (the gate) | 0.63 | 7.7% |
  | Stochastic masks | 0.34 | 2.9% |
  | Random masks (reference) | 1.54 | 22.7% |
  | Slice zeroed (reference) | 6.56 | 100% |

  - The gate metric had stayed between 0.070 and 0.119 since step 2,000
    (`docs/figures/e04/gate.png`). The final reading was 0.077.
- **Firing statistics.** 102,400 held-out tokens.
  - 599 of 7,680 components are alive (CI > 0.1 on at least 1 in 10^5 tokens).
  - Queries are nearly unused: 2 alive in layer 2 and 6 in layer 3.
- **Graph.** param-decomp's attribution harvester on 102,400 different held-out
  tokens, edge statistic |attr_abs|.
  - 600 nodes (599 components and the output) and 57,967 edges.
  - The 180 sources are layer 2's alive q, k, v, gate and up components.
  - Attributions ran in batches of 16 rather than 4: the same 800 rows in the
    same order, in a third of the time.
- **Causal test.** 16,384 held-out tokens, disjoint from both of the above.
  KL(target ‖ restricted) is measured over the full vocabulary.

## Q1: connected structure

![sieve curves](figures/e04/sieve_curves.png)

| Control | Peak z | Shuffle peaks: max / 99th percentile / median | Shuffles at or above the real peak | c* |
|---|---|---|---|---|
| Within-layer | 4.18 | 3.66 / 3.63 / 1.63 | 0 | 0.075 |
| In-strength | 4.46 | 7.30 / 3.96 / 1.61 | 1 | 0.075 |

- **Small c.** At small kept degree (c < 0.1) the graph connects slightly more
  interior nodes than its shuffles. This is the only region with positive
  learned connectivity.
- **Larger c.** From c ≈ 0.5 (within-layer) or c ≈ 0.8 (in-strength), the graph
  connects far fewer nodes than its shuffles. The lowest z is about −200
  (within-layer) and −110 (in-strength).
  - Attribution mass is concentrated on a few hub components, so the strongest
    edges keep returning to the same nodes.
  - e03's toy model showed the same split: connected structure at small c, hub
    concentration at large c.
- **Backbone.** Both sieves pick the same 12-component backbone at the same c*,
  all of it in layer 2:
  - attention: k[217], v[187], o[335];
  - MLP: gate[147, 203, 255], up[173, 488], down[121, 410, 539, 607].

  At this c the backbone is a shallow route: layer-2 components to the output,
  with no layer-3 component.

## Q2 and Q3: does the backbone carry behaviour?

![causal test](figures/e04/causal.png)

| Set (mask = CI × membership; rest of the model intact) | Components | KL (nats/token) | Share of the gap closed |
|---|---|---|---|
| None of the slice | 0 | 6.664 | 0% |
| Matched random sets, median (100 sets) | 12 | 6.656 | 0.1% |
| Matched random sets, best | 12 | 6.413 | 4.2% |
| Top mean-CI set | 12 | 6.398 | 4.5% |
| **Sieved backbone** | 12 | **6.317** | **5.8%** |
| All components | 599 | 0.710 | 100% |

The share of the gap closed is (KL_none − KL_set) / (KL_none − KL_all).

- The backbone beats all 100 matched random sets (Q2) and the importance-ranked
  set of the same size (Q3).
- In absolute terms it carries little: 2% of the alive components close 5.8% of
  the gap.

## What the data show (exploratory, not pre-registered)

**One attention pair does most of the work.** The random sets split into two
groups: 93 near 0% and 7 near 4%. All 7 contain both L2 v[187] and L2 o[335].
Sets with only one of them stay near 0%.

A follow-up measurement on the same rows (`experiments/e04_explore_pair.py`,
`results/e04/explore_pair.json`):

| Set | Share of the gap closed |
|---|---|
| v[187] alone | 0.0% |
| o[335] alone | 0.0% |
| v[187] + o[335] | 4.0% |
| Backbone without the pair | 0.5% |
| Backbone | 5.8% |
| Top mean-CI set without the pair | 0.5% |
| Top mean-CI set | 4.5% |

- **The pair is the smallest unit that does anything.** An attention output
  needs both a value direction and an output direction, and either alone
  changes nothing.
- **The backbone's other 10 components work together with the pair.** Alone
  they close 0.5%. With the pair they add about 1.3 points beyond the sum
  (5.8% against 4.0 + 0.5).
  - The top-CI set's other components only add up: 4.5% ≈ 4.0 + 0.5.
  - So the backbone beats picking by importance (Q3) through MLP components
    that are connected to the attention route. That is what the sieve was
    designed to find.
- **The Q2 null is lenient here.** Random sets are matched module by module, so
  they rarely contain both halves of the pair (7 in 100). The backbone still
  beats the 7 that do (best 4.2%), but most of its margin over the median
  random set comes from containing the pair.

## Replication (seed 1, pre-registered)

Run `p-542cdc24`: the same configuration with decomposition seed 1. The gate
addendum pre-registered it with the same gate and Q1-Q3 rules (R-Q1 to R-Q3),
before any seed-0 evaluation was read. It ran 06:37-12:39 on 2026-10-18 without
a restart. Results are in `results/e04_seed1/` and figures in
`docs/figures/e04_seed1/`.

- **Quality.** The CI-masked model is 0.65 nats (KL) from the target; zeroing the
  slice costs 6.55. It leaves 8.3% of the gap unrecovered, against 15% for
  random masks.
- **Graph.**
  - 873 of 7,680 components are alive (seed 0: 599).
  - 874 nodes, 113,554 edges, 274 sources.
- **R-Q1 fails under both controls.**

  | Control | Peak z | Shuffles at or above it | Shuffle max | p | c* |
  |---|---|---|---|---|---|
  | Within-layer | 2.96 | 6 | 4.85 | 0.07 | 0.020 |
  | In-strength | 3.16 | 3 | 4.47 | 0.04 | 0.097 |

- **Backbones.** All in layer 2.
  - Within-layer, at c* 0.020: 6 components, v[182], o[59], gate[22], up[84]
    and down[168, 251].
  - In-strength, at c* 0.097: 17 components, including those 6.
- **R-Q2 and R-Q3 pass, by 0.002 nats.**

  | Set | KL (nats/token) | Share of the gap closed |
  |---|---|---|
  | None of the slice | 6.664 | 0% |
  | Matched random sets, median | 6.663 | 0.01% |
  | Matched random sets, best | 6.435 | 3.83% |
  | Top mean-CI set | 6.435 | 3.83% |
  | **Sieved backbone** | **6.433** | **3.87%** |
  | All components | 0.686 | 100% |

**The attention pair again (exploratory).** The 5 random sets that close more
than 1% of the gap all contain both L2 v[182] and L2 o[59]. Both components are
also in the backbone and in the top mean-CI set. Pair check
(`experiments/e04_explore_pair.py --v 182 --o 59`):

| Set | Share of the gap closed |
|---|---|
| v[182] alone, o[59] alone | 0.0% each |
| v[182] + o[59] | 3.82% |
| Backbone without the pair | 0.21% |
| Backbone | 3.87% |
| Top mean-CI set without the pair | 0.005% |
| Top mean-CI set | 3.83% |

- **The two seeds' pairs look like the same mechanism.**
  - Value directions: firing rate 0.183 and 0.188, mean CI 0.155 and 0.151.
  - Output directions: firing on 97% and 99% of tokens, mean CI 0.94 and 0.99.

  It is an attention output active on nearly every token, through one value
  direction that is active on about a fifth of them.
- **Seed 0's finding that the backbone's MLP components work together with the
  pair does not replicate.** They added about 1.3 points in seed 0 and add 0.04
  in seed 1.

## Caveats

- **Two decomposition seeds, one slice, one model.**
- **The p-value floor is 0.0099 with 100 shuffles or random sets.** Seed 0's
  Q1 in-strength test failed by exactly one shuffle. More shuffles would
  estimate that p-value better, but the rule is "beats all 100". Seed 1 fails
  both controls by more than one shuffle anyway.
- **The alive threshold depends on token count.** At 10^-5 over 102,400
  tokens, a component must fire at least twice to count as alive (599 alive
  in seed 0, 873 in seed 1). A smaller sample would have admitted more.
- **The slice was chosen for the size of its effect** (zero-ablation KL ≥ 1
  nat; `e04_design.md`). Layers 2-3 sit early in the network, so the result may
  not hold for the middle layers, where the first choice (13-14) barely
  mattered to the output.

## Next

- **Find out what the always-on layer-2 attention pair computes.** It dominates
  the causal results in both seeds. Two candidates are an attention-sink route
  and a positional signal.
- **A causal null that keeps the pair's AND structure.** Per-module matching
  rarely draws both halves, so Q2's random sets are a weak baseline whenever
  one route dominates.
- **The sieve itself.** Its small-c excess was weak in seed 0 and absent in
  seed 1. On this slice, the connected structure it detects does not stand out
  from strength-matched noise.
