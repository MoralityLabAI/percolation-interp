# e03: percolation sieve over a full VPD and an SAE tower

Written 2026-10-11, before any e03 model, decomposition or SAE exists.

## Question

VPD (Goodfire's parameter decomposition) and SAEs carve the same model into
units in different ways: rank-one weight components versus activation
features. Each gives a cross-layer attribution graph. The questions:

- Which parts of each graph are connected structure rather than noise? This
  is the percolation sieve.
- Does the structure that survives the sieve correspond across the two
  methods better than the rest does?

## Pipeline

1. **Target.** A 2-layer LlamaSimpleMLP: d = 128, MLP 512, 4 heads, vocabulary
   4,019, about 1.4M parameters. It is trained offline on SimpleStories with
   param-decomp's own pretraining code (`configs/e03/pretrain_ss_llama_2L128.yaml`).
2. **Full VPD** with param-decomp (`configs/e03/vpd_ss_llama_2L128.yaml`):
   - every module (q, k, v, o, MLP up, MLP down) in both layers, with the
     delta component;
   - a global shared-transformer causal-importance (CI) function;
   - the paper's loss recipe, at 40k steps.
3. **VPD graph.** Run param-decomp's harvest, then dataset attributions:
   gradient × activation between components, summed over the dataset, with
   residual skip edges. Nodes are the alive components; edge weight is
   |attribution|. The layers are, in order:
   - token embedding;
   - layer-0 attention;
   - layer-0 MLP;
   - layer-1 attention;
   - layer-1 MLP;
   - output.
4. **SAE tower.**
   - TopK SAEs on the residual stream at three levels: after the embedding,
     after block 0, and after block 1 (before the final norm).
   - 1,024 latents each, k = 16, with an auxiliary loss on dead latents.
   - Per-token TopK, not BatchTopK (implementation decision, taken before any
     SAE was trained): "a feature fires" is then exactly "it is in this
     token's top k".
5. **SAE graph.**
   - Feature-to-feature edges between levels, using the same construction as
     param-decomp's dataset attributions. For each target feature j, backpropagate
     the batch sum of its activation to the source level's residual stream, and
     read |a_i · (d_i · grad)| per token. a_i is the source feature's activation
     and d_i its decoder direction.
   - Edges are summed over tokens, including the skip edge from level 0 to
     level 2.
6. **Cross-reference** between components and features, on the same tokens:
   - **Co-firing.** A component fires when its CI > 0.1; a feature fires when it
     is in the top-k. Association is PMI with Jaccard as a secondary measure.
   - **Weight space.** Cosine between a component's read direction (V column of
     q, k, v or MLP-up in layer l) and level-l SAE encoder directions. Cosine
     between a component's write direction (U row of o or MLP-down in layer l)
     and level-(l+1) SAE decoder directions.
7. **Sieve.**
   - Each graph gets per-layer ranking in degree units (c), a within-layer
     shuffled control, and learned connectivity: the backbone node fraction
     minus the shuffled control's, across c.
   - c* is the occupation where learned connectivity peaks.
   - The sieved structure is the backbone at c*.

## Predictions

The VPD graph and SAE graph are fixed before these are scored.

**P1: both graphs carry learned connectivity.**
- The maximum learned connectivity over c is above the 99th percentile of the
  same quantity across 20 re-shuffles.
- Scored separately for the VPD graph and the SAE graph.

**P2: cross-method agreement concentrates on the sieved structure.**
- Statistic: for each VPD backbone component, its best co-firing PMI with any
  SAE backbone feature in the matching levels; then the mean over components.
- That mean exceeds the same statistic for random node sets. Random sets match
  the backbone's size per layer and come from the same layers; for features
  they are also matched on firing-rate decile.
- **Test:** 1,000 random sets, one-sided p < 0.01.
- **Effect size reported:** the ratio of the backbone statistic to the median of
  the random sets.

**P3: the same, in weight space.**
- As P2, with best cosine (read and write directions) in place of co-firing
  PMI.
- One-sided p < 0.01 against the same matched random sets.

**Reading the result:**
- **P2 and P3 both hold:** the sieve isolates structure that the two
  decompositions agree on.
- **P2 holds, P3 does not:** they agree on when units are used, not on
  directions.
- **Neither holds:** percolation-connected structure is method-specific.

## Controls and pitfalls

- **Dead units.** Components that never fire, and features that never fire,
  are dropped before any graph is built, using param-decomp's alive masks and
  a firing count of at least 1 in 10^5 tokens for SAEs.
- **Frequency.** Frequent units co-fire with everything. That is why random
  sets are firing-rate matched and PMI is the primary association measure.
- **Choice of c\*.** It is picked from learned connectivity alone, never from
  cross-reference results.
- **Sign.** Graphs use |attribution|. Signed attributions are kept for
  follow-up.
