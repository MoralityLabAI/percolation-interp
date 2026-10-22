# E08 prototype: holonomy percolation / Gödel Globe

**Status: engineering calibration only.** This was written after E07 had been inspected, and is not a pre-registered experiment or evidence of semantic polysemanticity. The repository's `master` branch did not yet contain the local E07 geometry work when E08 was developed. E06's registered controls, criteria and results are untouched.

## What is implemented

`src/percinterp/cycle_filtration.py` runs one nested undirected bond-percolation sweep over explicitly provided `src`, `dst`, and entry keys in `[0,1]`. Its output has the largest cluster as a fraction of designated component vertices, the first Betti number `β₁ = |E_p| - |V| + c(p)`, and fundamental loops formed as Kruskal forest edges are inserted. Stable ordering resolves ties. Parallel contacts can generate two-edge loops.

`src/percinterp/holonomy.py` estimates an orthogonal connection from local orthonormal feature frames:

    Q_(u→v) = polar(U_v^T U_u)
    H_γ = Q_(last edge) ... Q_(first edge)

For a reversed edge, its transpose is used. The smallest singular value of frame overlap gates edge certification. Uncertified edges still enter *structural* percolation curves, but a separate cycle filtration of certified edges determines holonomy. Thus the certified graph cannot accidentally close a loop through an unreliable link.

Under local gauge changes `U_i -> U_i R_i`, each edge map changes as `Q_uv -> R_v^T Q_uv R_u`; closed-loop matrices change by conjugation. Eigenphases, determinant and Frobenius distance `||H-I||_F` remain unchanged. A signed 2D rotation is gauge-dependent under orientation-reversing frame changes; its **magnitude** is invariant for SO(2) loops. For rank 1 there are only ±1 signs. Rank 3 supports a rotation angle; ranks 4+ export eigenphases instead.

The first threshold supporting `||H-I||_F > ε` among certified fundamental loops is recorded as `p_h`. This is exact if a qualifying loop was encountered before the record cap, or if all loops were inspected. When a cap truncates before any qualifying loop, `p_h=null` and `p_h_known=false`. `p_h` depends on the chosen frame estimator and certification rule. The mathematical connection is extra structure beyond the graph's degree sequence.

`experiments/e08_holonomy_percolation.py` generates a synthetic six-node **bipartite** fixture with two cycle births, `.37` and `.68`. Its `planted` mode inserts SO(2) holonomies of magnitude 45° and 90°. `flat` uses identity transports, and `tangent` computes the transport from tangent frames on a sphere. The fixture is unrelated to Qwen activations. It exports JSON or a self-contained JS dataset for the browser.

`ui/godel_globe/index.html` works offline and with no JS dependencies. The bond slider reveals the nested graph and its β₁ curve; selecting a loop replays a courier along actual edge transports. The reference needle stays fixed as the transported compass accumulates the measured 2D rotation. Dragging the globe changes only the camera. The visual sphere layout of the six nodes is a display embedding, not an estimated latent manifold.

## Quick start (CPU only)

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_holonomy_percolation.py
python experiments/e08_holonomy_percolation.py --mode planted
```

Open `ui/godel_globe/index.html` locally. For a distinct calibration input use:

```bash
python experiments/e08_holonomy_percolation.py --mode tangent --output out/e08_tangent.json
python experiments/e08_holonomy_percolation.py --mode flat --output out/e08_flat.json
```

## How E07 can connect, once pushed

1. Turn an individual E07 coordinate map's top-*k* component-to-neuron contacts into global `src,dst` arrays: component nodes `0..C-1`, coordinate nodes `C..C+N-1`. Preserve the distinction between independent Bernoulli occupation keys and deterministic top-*k* edge selection.
2. Derive shared-rank orthonormal local frames from **activation-conditioned** VPD subspaces. Their node correspondence and common ambient space must be justified. A single rank-one component cannot produce continuous SO(2) holonomy. Extract `Q` with `frame_edge_transports` and retain the overlap certification mask.
3. Run `analyze_connection` with fixed occupation keys, `n_component_vertices=C`, transport matrices and a fixed ε. Use enough cycle-record capacity to certify `p_h`, and report truncation and uncertified-edge fraction.
4. Rebuild **matched null geometries** after degree-preserving rewiring. Reusing learned edge transports across a rewired topology would create a meaningless control. Keep cycle-length and overlap distributions under inspection. Repeat across decomposition seeds without aligning component IDs naively.
5. Only after calibrations and null controls: pair loop signatures with signed CI-weighted token contexts, intervention directions and causal readouts. E07's structural connectivity alone does not identify semantic features.

E07 reports that same-degree rewiring reproduces cluster curves within about 0.002 at the 50% crossover, while the eight selected contact coordinates cover roughly 2.72–3.64% of median direction energy. Holonomy analyses need explicit feature-frame construction and stronger nulls before attaching semantic interpretations.

## Calibration checks

`tests/test_holonomy_percolation.py` checks exact cycle ranks against brute force; correct birth keys and signed traversals; 45°/90° planted loops; flat gauge changes; curved gauge invariance; unreliable edges; truncation; rank-one signs; flat-frame Procrustes and spherical tangent-frame curvature. It uses only numpy/pytest and touches no GPUs, paid pods or saved research outputs.
