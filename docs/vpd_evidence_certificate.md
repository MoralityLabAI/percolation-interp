# Optional VPD evidence receipt review (v1)

This standalone CPU-only validator reads a JSON receipt and artifacts from disk. It does not import Goodfire or invoke model inference, alter existing graphs, or touch E08/E10 results.

Primary scientific reference: [Goodfire parameter-decomposition handbook](https://github.com/goodfire-ai/param-decomp/blob/main/docs/handbook.md). Source index: [Goodfireverse](https://github.com/patrickdugan/Goodfireverse). Its evidence standard distinguishes unmasked parameter parity, reconstruction under a named output metric, training convergence, minimality relative to a domain-specific anchor, and fresh adversarial-mask performance. Each dimension is independent.

## Usage

```bash
python -m percinterp.vpd_evidence certificate.json --artifact-root ./artifacts
python -m pytest -q tests/test_vpd_evidence.py
```

A receipt has `protocol: {id, source, version, criteria}`, `artifacts: {name: {path, sha256}}`, and `observations: {dimension: measurement}`. The five supported dimensions are `unmasked_parity`, `reconstruction`, `convergence`, `minimality`, and `adversarial_mask`. Each protocol criterion must supply its *own* `source`, an exact `metric`, `operator` (`<=` or `>=`) and numerical `limit`. A measurement supplies the matching metric, numerical value and the name of a bound artifact. A SHA-256 digest is verified against bytes in the given root. Reusing a digest or changing an artifact causes the receipt to fail integrity checking.

Mask policy requirements: unmasked parity must declare `all_ones_delta_excluded`; reconstruction declares `importance_values`, `rounded_importance` or `random`; adversarial evaluation requires `fresh_optimized_adversarial`, distinct evaluation/training run identifiers, `shared_across_batch: true`, and a nonempty `budget` object describing the attack. A random-mask or training-adversary test leaves adversarial evidence **unknown**.

Convergence requires a numerical multi-point `settled_window` and a study-defined drift criterion. Minimality requires a named `anchor` and a protocol criterion calibrated to that domain. The CLI exit code is 2 on invalid evidence or failed artifact integrity. A measured failure of a declared criterion is reported in the JSON; consumers must check each dimension, not only the process exit code.

## Interpretation and limitations

**No output of this tool is scientific certification.** The `scientifically_certified` field is always `false`. The validator cannot independently replay model forward passes, recalculate parity of weight matrices, establish that a loss was computed on an independent holdout, verify data-distribution coverage, verify convergence from training logs, establish global minimality, or confirm that a reported adversarial attack actually ran. Producer-supplied `source`, metric, budget, window, mask policy, run IDs and anchor are declarations. Hashes establish byte consistency for locally supplied artifacts; they do not establish authenticity, independent replication, or scientific soundness.

A comparison marked `meets_stated_criterion` means only that the recorded value meets the criterion included with the receipt. It does not endorse the study protocol, verify that its threshold was preregistered, or imply that it is a universal cutoff. Missing evidence is `unknown`. Metric mismatches and broken artifact binding are `invalid`. Synthetic fixture values and thresholds test software behavior and have **no empirical significance**.

The handbook specifically warns that fresh, shared-batch adversarial evaluation is important, that random masks test only average-case behavior, and that gradients can find worse mask combinations. A receipt can record that procedure, but it cannot attest the computation. The evaluator's attack strength, data and downstream interventions must be reviewed scientifically. Reconstruction under causal-importance masks is monitor-only evidence, rather than an adversarial test. Attribution-graph pruning requires its own downstream adversarial check.

This v1 deliberately avoids reading any existing E03/E08/E10 evidence as qualified, changing Goodfire's producer contracts, or defining universal KL, MSE, sparsity, parity or convergence thresholds.
