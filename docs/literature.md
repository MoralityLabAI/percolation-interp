# Prior-art check (October 2026)

Two search passes: one for direct prior art, one for adjacent literatures.

- **[V]** checked against the abstract or paper text.
- **[S]** seen only in search snippets.

"No prior work found" means the searches turned nothing up. It is not proof
there is none. Before writing anything up, check Google Scholar citations of
Pesce et al., Devlin & Sanders and MIB, plus LessWrong and the Alignment Forum.

## Verdict on the gap

It appears open. No paper found treats a trained network's causal or attribution
graph as a percolating system: none measures spanning, backbone, largest
component or bottleneck against the kept fraction p and ties them to circuit
fidelity, against shuffled and at-init controls, across widths and checkpoints.
Each neighbour holds one piece:

| Work | Has | Lacks |
|---|---|---|
| Pesce et al. 2026 | trained graphs, a p sweep, critical exponents, random controls | causal graphs (uses weight magnitudes); fidelity is accuracy after retraining; no connectivity observables |
| Devlin & Sanders 2025 | spanning, width/depth scaling | trained weights (random dropout masks only) |
| MIB, Edge Pruning, Hanna et al. | faithfulness against edges kept | any critical or connectivity analysis |
| Tamai et al.; Ghavasieh et al. | directed-percolation universality, finite-size scaling | graph structure (they study signal or activity dynamics) |
| Lubana et al. 2025 | percolation tied to capability emergence | a model graph (theirs is built from the data) |

**Main novelty risk.** The faithfulness-versus-edges curve already exists as a
standard metric (MIB's CPR/CMD). The contribution has to rest on the
connectivity observables, the null models, and the finite-size and checkpoint
scaling.

## Closest work: cite and distinguish

**Pesce, He, Caldarelli et al., "Phase Transitions in Neural Networks Pruning"** (arXiv 2602.15224, Feb 2026). [V]
- Iterative magnitude pruning plus fine-tuning on trained MLPs and CNNs (MNIST,
  KMNIST), with edge density as the control parameter.
- Accuracy and prediction entropy fit power laws in (p − p_c).
- Controls are random edge subsets and degree-preserving rewiring.
- No spanning or giant component, no width scaling, no causal graph.

**Devlin & Sanders, "Dropout Neural Network Training Viewed from a Percolation Perspective"** (arXiv 2512.13853, v3 Sep 2026). [V]
- DropConnect is modelled as bond percolation and dropout as site percolation
  on layered, fully connected graphs, with random removal only.
- Crossing probability goes to 0 when width grows slower than log(depth).
- A non-trivial p_c exists at width ≈ c·log(depth).
- Without an input-output path, bias-free networks get zero gradient.
- Use: an analytic null. Our bond-percolation runs (e01, A1) show the related
  ln N / L control.

**Lubana, Kawaguchi, Dick, Tanaka, "A Percolation Model of Emergence: Analyzing Transformers Trained on a Formal Language"** (ICLR 2025, arXiv 2408.12578). [V]
- Bond percolation on a bipartite entity-property graph of the training data,
  with p_c ≈ 1/√(|E||K|).
- Closest name collision. Its graph is the data, not the model.

**Brill & Ingebretsen Carlson, "Critical Percolation as a Synthetic Data Model for Interpretability"** (arXiv 2606.20347, June 2026). [V]
- Builds synthetic data from hierarchical functions on critical percolation
  clusters, then probes residual MLPs trained on it.
- Percolation is the data model, not the analysis. Background.

**Tamai, Okubo, Duy, Natori, Todo, "Universal Scaling Laws of Absorbing Phase Transitions in Artificial Deep Neural Networks"** (Phys. Rev. Research 7, 033072, 2025; arXiv 2307.02284). [V]
- Signal propagation at initialisation, near the edge of chaos.
- MLPs fall in the mean-field class and CNNs in the directed-percolation class,
  with finite-size scaling tied to the depth-width trade-off.
- Background for the universality-class question.

**Ghavasieh et al., "Toward a Physics of Deep Learning and Brains"** (arXiv 2509.22649, 2025). [V]
- Activity avalanches in trained networks show directed-percolation and
  Barkhausen classes, and susceptibility predicts learning.
- Dynamics, not influence graphs. Background.

**Minegishi, Iwasawa, Matsuo, "grokked tickets"** (TMLR, arXiv 2310.19470). [V]
- Grokked lottery tickets have longer path lengths and lower clustering, and
  their structure changes quickly near generalisation.
- Closest competition for the grokking arm.

## Pruning and connectivity

- **SynFlow** (Tanaka, Kunin, Yamins, Ganguli, NeurIPS 2020, arXiv 2006.05467). [V]
  - Defines "layer collapse" (pruning a whole layer, which disconnects the
    network) and critical compression.
  - Its score is each weight's share of the l1 path norm.
  - Must cite: layer collapse is a spanning failure on the weight graph.
- **Vysogorets & Kempe, "Connectivity Matters"** (JMLR 24, 2023, arXiv 2107.02306). [V abstract]
  - Effective sparsity counts weights cut off from any input-output path.
    Randomly pruned networks have effective compression far above nominal.
  - Report effective p (edges on the backbone) beside nominal p. Our
    `backbone_edge_fraction` computes it.
- **Hoang et al.** (ICLR 2023): a Ramanujan-graph view of pruning at initialisation. [S]
- **Filan et al., "Clusterability in Neural Networks"** (arXiv 2103.03386). [S]
  - Compares trained weight graphs against layer-wise shuffled controls, which
    is precedent for our null.

## Grokking as a transition

- **Nanda et al., "Progress measures for grokking via mechanistic interpretability"** (ICLR 2023, arXiv 2301.05217). [V]
  - Memorisation, then circuit formation (test and train loss flat), then
    cleanup (test loss drops suddenly and weight norm falls).
  - "The circuit is formed well before grokking occurs."
  - Restricted and excluded loss are reusable fidelity references.
- **Rubin, Seroussi, Ringel** (ICLR 2024, arXiv 2310.03789). [V]
  - Grokking is a first-order transition in an adaptive-kernel analysis.
  - Keep that training-time transition distinct from our threshold-sweep
    transition.
- **Žunkovič & Ilievski** (JMLR 25, 2024, arXiv 2210.15435). [V]
  - Solvable models with exact critical exponents for grokking.
- **Liu, Michaud, Tegmark, "Omnigrok"** (ICLR 2023, arXiv 2210.01117). [V]
  - Grokking tracks weight norm.
  - This confounds absolute thresholds across checkpoints, since norms shrink
    in cleanup.
- **Merrill et al.** (arXiv 2303.11873): grokking as competition between a sparse and a dense subnetwork. [V]
- **Clauw et al.** (arXiv 2408.08944): grokking as an emergent transition, measured by synergy. [V]

## Circuit discovery and faithfulness curves

- **ACDC** (Conmy et al., NeurIPS 2023, arXiv 2304.14997). [V]
  - Sweeps its threshold τ and publishes ROC curves and KL against edge count.
  - Tasks: IOI, Greater-Than, Docstring, tracr.
  - Its scores depend on pruning order, so they are a poor fixed weighting to
    threshold.
- **Edge Pruning** (Bhaskar et al., NeurIPS 2024, arXiv 2406.16778). [V]
  - Learned L0 masks, faithful down to tiny edge fractions, scaled to 13B
    parameters.
  - Gives one binary mask per sparsity, not a score ranking.
- **EAP** (Syed, Rager, Conmy, BlackboxNLP 2024, arXiv 2310.10348). [S]
- **EAP-IG** (Hanna, Pezzelle, Belinkov, "Have Faith in Faithfulness", COLM 2024, arXiv 2403.17806). [V]
  - Node overlap is not faithfulness.
  - One missing input-to-MLP0 edge collapses faithfulness, which is an
    anecdotal connectivity bottleneck.
  - **Best tool to reuse:** github.com/hannamw/EAP-IG scores every edge on
    TransformerLens models.
- **Miller, Chughtai, Saunders** (COLM 2024, arXiv 2407.08734). [V]
  - Faithfulness shifts with ablation type, node versus edge ablation, token
    positions and aggregation.
  - Fix these choices, and report how the threshold moves when they change.
  - Code: github.com/UFO-101/auto-circuit (per-edge prune scores).
- **MIB** (Mueller et al., ICML 2025, arXiv 2504.13151). [V]
  - CPR and CMD are areas under faithfulness against edge fraction.
  - The standard for our fidelity axis, and the main overlap risk.
  - Code: github.com/aaronmueller/mib.
- **Circuit Tracing** (Ameisen et al., Anthropic, 2025). [V]
  - Prunes attribution graphs by cumulative influence, with hand-set
    thresholds.
  - Shows a size-against-completeness frontier for one prompt.
  - No connectivity analysis.
- **InterpBench** (Gupta et al., NeurIPS 2024, arXiv 2407.14494): models trained with known circuits. [S]
  - IOI, Greater-Than and Docstring circuits were found by hand, so they are
    not true ground truth.
- **Tigges et al.** (arXiv 2407.10827): EAP-IG across Pythia checkpoints. [S]
- **Du et al.** (ACL 2023, arXiv 2307.03056): semiring backprop finds highest-weight paths. [S]
  - Related to our bottleneck computation.
- **Ferrando & Voita, "Information Flow Routes"** (EMNLP 2024, arXiv 2403.00824): attribution-thresholded graphs. [S]
- **Deng et al.** (ACL 2026, arXiv 2506.18141): connected components of a thresholded SAE co-activation graph. [S]
  - No sweep and no null.
- **Tan et al., "Can Graph Learning Learn Circuits?"** (arXiv 2608.08536, Aug 2026): GNNs localise circuit edges. [V]
  - Network-science people working on circuits.

## Assumptions the plan relied on

- **Toy Models of Superposition** (Elhage et al. 2022). [V, pp. 17-20]
  - With uniform importance, features form polytopes in orthogonal subspaces
    ("many of the Thomson solutions can be understood as tegum products").
  - The paper's own feature-geometry graph is disconnected exactly across tegum
    factors.
  - So the original step 1 (does the interference graph percolate at the phase
    change?) has a known answer. It was replaced by the control pipeline.
- **Explosive percolation is continuous.** Riordan & Warnke (Science 333:322,
  2011 [S]; proof in Ann. Appl. Probab. 22:1450, 2012, arXiv 1102.5306 [V]).
  - Every Achlioptas process has a continuous transition.
  - Adding edges in trained-weight order is a correlated ordering, not an
    Achlioptas process.
  - A sharp jump still needs finite-size scaling before it is called
    discontinuous.
