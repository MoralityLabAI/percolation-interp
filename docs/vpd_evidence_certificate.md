# Optional VPD evidence-certificate review (v1)

The VPD receipt validator is an offline, Python-standard-library-only checker for **record consistency**. It executes no checkpoints or model inference. It does not modify PercInterp's E08/E10 protocols, graph importer, experiments, or frozen results.

Primary scientific authority: [Goodfire's parameter-decomposition handbook](https://github.com/goodfire-ai/param-decomp/blob/main/docs/handbook.md). Public source index: [Goodfireverse](https://github.com/patrickdugan/Goodfireverse).

## Run the validator

Run without installing PercInterp dependencies:

~~~bash
python src/percinterp/vpd_evidence.py certificate.json --artifact-root ./artifacts
~~~

Or from an installed package: `python -m percinterp.vpd_evidence certificate.json --artifact-root ./artifacts`.

The receipt is JSON with these fields:

- `schema`: exactly `vpd-evidence-certificate/v1`.
- `protocol`: nonempty `id`, `source`, `version`, plus `criteria`. Each criterion supplies its own nonempty `source`, exact `metric`, comparison `operator` (`<=` or `>=`) and finite numerical `limit`. These limits must come from a **named study protocol** before evaluation. The validator cannot look up or authenticate that protocol.
- `artifacts`: a name-to-`{path, sha256}` mapping. Paths are relative to `--artifact-root`, traversal outside the root is rejected, artifacts must contain JSON and be at most 16 MiB. SHA-256 is checked against exact file bytes.
- `observations`: each independently optional dimension has a `metric`, finite numerical `value` and dimension-specific metadata. Every supplied observation also names a verified `artifact` and an RFC 6901 `record_pointer` to the **complete observation record** inside that artifact. The record must equal the observation object after removing the linking fields `artifact` and `record_pointer`.

A minimal illustrative measurement artifact can contain:

~~~json
{
  "observations": {
    "reconstruction": {
      "metric": "kl_per_position",
      "value": 0.07,
      "mask_selection": "importance_values"
    }
  }
}
~~~

The receipt pointing to that artifact includes its exact SHA-256 under `artifacts.run`, along with this observation:

~~~json
{
  "artifact": "run",
  "record_pointer": "/observations/reconstruction",
  "metric": "kl_per_position",
  "value": 0.07,
  "mask_selection": "importance_values"
}
~~~

The sourced criterion for this observation must use the same `kl_per_position` metric. This example is schematic and its number has no empirical significance.

## Dimensions and evidence requirements

| Dimension | Required recorded evidence | Interpretation |
|---|---|---|
| `unmasked_parity` | `max_abs_parameter_error` under `mask_selection: all_ones_delta_excluded` | Parameter-sum parity criterion, declared by producer |
| `reconstruction` | Study-specific, exact-match target-output metric; `importance_values`, `rounded_importance` or `random` mask policy | Reconstruction under named mask family; monitor-only masks remain monitor-only |
| `convergence` | `settled_window_drift`, `annealing_complete: true`, at least three finite settled-window values and strictly increasing `step_indices` | Compute drift as `max(settled_window) - min(settled_window)` against a protocol limit |
| `minimality` | `active_subcomponents_per_input` and a nonempty domain-specific `anchor` | Recorded active-count comparison against that domain's baseline |
| `adversarial_mask` | `mask_selection: fresh_optimized_adversarial`, distinct nonempty `training_run` and `evaluation_run`, nonempty attack `budget`, `shared_across_batch: true` | Recorded fresh shared-batch adversarial result under a study-specific metric and limit |

For convergence, the reported drift must agree numerically with the recorded window range. Incomplete annealing or insufficient history remains **unknown**. A well-formed window with drift above the protocol threshold is a **recorded failure**, while a contradictory claimed drift is **invalid**. The minimum window length is a format sanity condition, not proof of true convergence, which requires evaluating settled loss and sparsity curves and gamma schedules.

The handbook identifies fresh adversarial masking as essential because random masking does not test deliberately chosen joint ablations. A random-mask record or a reused training adversary is **unknown** for the adversarial dimension. Mask-selection labels, run identifiers and budget fields are producer assertions.

## Output and decision policy

Each dimension reports one of:

- `unknown`: omitted/incomplete evidence, missing criterion, absent domain anchor, unfinished annealing, or random-mask evidence submitted as adversarial evaluation.
- `invalid`: contradictory or malformed evidence, metric disagreement, wrong parity policy, or an artifact record disagreeing with the receipt.
- `meets_stated_criterion` / `fails_stated_criterion`: arithmetic comparison of artifact-bound, *producer-recorded* numbers against the identified protocol limit.

`artifact_integrity` is `verified` only when the receipt structure and all listed artifact bytes pass validation. The CLI exits with code 2 for invalid dimensions or failed integrity, and 0 for structurally acceptable receipts, **including records that fail scientific criteria**. Inspect the dimension statuses; exit code 0 is not a scientific success signal.

`scientifically_certified` is **always false**. Even five `meets_stated_criterion` records do not produce an overall qualification decision.

## What cannot be established here

This tool cannot independently:

- Reconstruct parameter sums or run the all-ones decomposition to establish parameter or behavioral parity.
- Recompute target KL/MSE/accuracy or any custom output metric; establish held-out data independence or distribution coverage.
- Verify training-step history, gamma annealing, genuinely settled convergence, a sparsity sweep or suitable dense, ground-truth and architecture-matched minimality baselines.
- Reproduce an adversary, verify its mask search/budget/freshness/shared-batch procedure, assess attack strength, or test all combinations of partial ablations.
- Verify that source hashes, run identifiers, JSON records and protocol strings came from an independent scientist. Producers can submit self-consistent but fabricated measurements. Hashes bind bytes, never authenticity.
- Establish subcomponent semantics, causal mechanism identity, graph pruning robustness, parameter faithfulness, real-model effectiveness or independent replication.

Goodfire's full standard requires native parameter-faithfulness checks, reconstruction/minimality Pareto fronts after convergence, and held-out adversarial tests. An attribution graph is not itself a verified computational mechanism. Scientific conclusions require separately reviewed measurements.

## Synthetic test contract

~~~bash
python -m pytest -q tests/test_vpd_evidence.py
~~~

Deterministic fixtures cover sparse-but-unfaithful, dense-but-faithful, unconverged, metric-mismatched, random-mask and stale-adversary cases; tests also check path confinement, artifact corruption, mismatched artifact records, missing evidence, schema errors and the standalone CLI. Their thresholds and values are exclusively synthetic. The separate read-only `vpd-evidence.yml` GitHub Actions workflow exercises these tests on Python 3.11, 3.12 and 3.13 without model execution.
