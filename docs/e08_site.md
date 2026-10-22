# E08 website: staged Gödel Globe

This is a static site, served by GitHub Pages from master. The payload is a small
HTML/JS bundle without runtime network calls, CDN, GPU, model weights or credentials.

## Three calibration fixtures

The selector offers three synthetic SO(2) connections over the same six-node
component/coordinate graph and identical entry keys:

- planted: known 45° and 90° loop rotations;
- flat: identity transports and trivial holonomy;
- tangent: orthogonal transports estimated from frames tangent to an illustrative sphere.

The structural percolation curves match across all modes; their transport differs.
The globe's embedding is illustrative, not a measured feature manifold.

## E07 measurement shelf

The website separately displays the reported 50%-largest-component crossover
values from the saved 2026-10-22 E07 exploratory Qwen3-0.6B study. Each row
represents two decomposition seeds of one target model. Learned, same-degree
rewired and uniform-contact p50 values are:

| Role | Learned seeds 0/1 | Same degree seeds 0/1 | Uniform seeds 0/1 |
|---|---|---|---|
| L0 gate write | 0.151 / 0.153 | 0.152 / 0.153 | 0.326 / 0.331 |
| L0 up write | 0.303 / 0.300 | 0.303 / 0.301 | 0.326 / 0.329 |
| L0 down read | 0.307 / 0.305 | 0.307 / 0.303 | 0.326 / 0.328 |
| L1 gate write | 0.231 / 0.228 | 0.230 / 0.227 | 0.325 / 0.325 |
| L1 up write | 0.177 / 0.177 | 0.178 / 0.175 | 0.327 / 0.327 |
| L1 down read | 0.222 / 0.222 | 0.223 / 0.222 | 0.328 / 0.326 |

These are finite-graph 50% crossover points rather than critical p_c estimates.
Same-degree rewiring reproduces the learned cluster curves closely. The top-eight
contact coordinates carry about 2.72–3.64% of a median direction's squared
energy. Neither feature semantics nor model-specific holonomy were measured.

The globe continues to use synthetic fixtures. Real E07 component-neuron
contact arrays and activation-conditioned frames have not been loaded into it.

## Build

    python -m pip install -e . pytest
    python -m pytest -q tests/test_holonomy_percolation.py tests/test_e08_site.py
    python experiments/e08_build_site.py --output out/e08_site
    python -m http.server 8000 -d out/e08_site

Open http://localhost:8000 and select each fixture. The site also works locally
from file:// once built.

## GitHub Pages

Deployment workflow: .github/workflows/percinterp-pages.yml.

First select repository Settings → Pages → Build and deployment → Source:
GitHub Actions. The first-time setting is required before configure-pages and
deploy-pages can publish. Once the Pages deployment succeeds, the site is at:

https://moralitylabai.github.io/percolation-interp/
