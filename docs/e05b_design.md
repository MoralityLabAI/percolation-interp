# e05b: e05 with a 16× larger decomposition batch, trained on a rented GPU

Written 2026-10-21, before any e05b decomposition existed and before any e05 Q1-Q4
quantity was computed. Neither e05 seed was scored on Q1-Q4:
- seed 0 failed the gate at 20k steps (KL 2.84 against the 1.94 limit);
- seed 1 was cancelled at step 9,815, before its gate.

## Why

e05's decompositions are the local 16 GB card's best:
- batch 4 × 128 tokens;
- quality plateaued from step ~12,000, at a CI-masked KL of about 2.6-2.9.

The suspected limit is batch size. Gradient accumulation cannot substitute for it,
because param-decomp no longer supports it. Memory grows about 2.7 MB per token, so a
larger batch needs a larger GPU.

## What changes from e05

**Only the decomposition's batch size: 64 × 128 tokens instead of 4 × 128.**
- That is 16× more tokens per step: 164M over 20k steps, about 12 passes over the
  pile-10k training split (e05 saw about 0.7).
- The data is otherwise unchanged. Overfitting would show in the gate, which is
  measured on held-out rows.

**Hardware.** It trains on a rented 48 GB GPU (Prime Intellect, an A6000 or
equivalent), because the estimated peak is about 31 GB.

**Budget rule, fixed now.** A short benchmark on the pod measures the step time first.
- If 20k steps at batch 64 would cost more than $10 per seed, the batch drops to 32.
- Nothing else is adjusted.
- The total cap is the $25 Prime balance.

**Everything else is e05, unchanged:**
- the slice (layers 0-1);
- the component counts and loss recipe;
- learning rate and schedule, and 20k steps;
- seeds 0 and 1;
- the gate (KL(target ‖ CI-masked) ≤ 0.25 × KL_PAMA);
- Q1-Q4 and their rules;
- the analysis code, which runs locally on the returned decompositions.

## If the gate fails

Each seed is scored only if it passes its gate. If both seeds fail at batch 64, e05b
reports that VPD at this budget does not decompose layers 0-1 beyond their
constants. No further retrains are pre-registered.
