# QPR-01: planted-circuit recovery, prospective v1

This separate study leaves E10 and its owners untouched. The baseline inspected
for this addition is master `9f6cc2b8fbda963e5156863709cc044d16a1e725`.
`recovery.py` supplies exact scoring and rejection checks. It runs no selector,
loads no model, and establishes no incremental benefit for percolation.

## Registered comparison

Freeze the Qwenputer source commit, checkpoint hashes, runtime, dtype, hardware,
plant generator and answer-key digest before collection. They remain UNVERIFIED
here; author reports are not our replications. A separate custodian keeps the
answer key hidden from selectors. Calibration seeds are 0..15, held-out seeds
16..47. Use planted exact memory/parity controls, sham plants and benign changes.
The exact rational controls in EvalMath supply a first oracle.

Compare percolation with degree, weighted degree, mean causal importance,
attribution mass, firing rate, frequency-matched random, and a plain random
negative control. SAE/dictionary, local eigenmode and VPD analyses are separate
representation arms crossed with selectors where the representation is valid.
For each arm publish transform hash, reconstruction residual and dimension.

Primary endpoint: held-out causal task restoration after selective ablation and
rescue at k=4 recovered components. Secondary quotas are k=1,2,8. Match candidate
universe, retained component count, forward-pass/patching budget, seed, inputs and
intervention sites. Thresholds 0.1, 0.25 and 0.5 are calibration-only choices;
select one by a preregistered calibration criterion before revealing held-out
results. Do not optimize on test seeds or change E10 thresholds.

First reproduce the unrotated positive control. Then jointly transform the known
memory state, keys, queries and decoder, verifying trace equality before scoring.
A transformed-state-only corruption is a negative control. Full-model rotations
require a separately verified architecture-preserving map. Use subspace overlap
with matched ranks after rotation; neuron-index overlap has different semantics.

Run recurrent swaps with attention intact and with attention intervention, plus
attention-only intervention and sham/rescue arms. Record task behavior separately
from probe scores. Include redundant mechanisms so single-site null effects can
be interpreted against a known answer key.

## Decision rule

H0: percolation has no positive paired held-out restoration gain over the strongest
prespecified baseline at k=4. H1: the gain is positive. Freeze a seed-clustered
paired bootstrap before collection: 10,000 resamples, RNG seed 20261028; in each
resample compare against the maximum baseline mean. Acceptance requires the 95%
percentile lower endpoint above zero, passing positive controls, and matched costs.
A nonpositive point estimate overturns the directional incremental-benefit claim
for this experiment. A strictly nonpositive upper endpoint supplies stronger
statistical evidence against it. An interval spanning zero is inconclusive.
Missing artifacts, failed reconstruction/positive controls, label leakage or
unmatched budgets invalidate the experiment. The scorer currently emits descriptive
paired summaries; the bootstrap and real-model adapters remain unimplemented.

Run the nine local scoring tests from the repository root:

    python -m unittest discover -s research/qwenputer_v1 -p 'test_*.py' -v

## Sources and companion work

[Upstream compiled-circuit experiment](https://divagr.com/blog/building-oversight-tools-into-a-models-weights),
checked 2026-10-28, motivates the recovery target. Its own author identifies the
plant as a positive control and proposes rotated coordinates as a harder test.
[EvalMath](https://github.com/MoralityLabAI/EvalMath) owns the exact transition
contract. Compactification-Theory has the abstraction note, with publication
status recorded in the cross-repository handoff. BlueBeam and Control-Harness
cover audit integrity and host authorization respectively.
