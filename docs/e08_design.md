# e08: from coordinate sharing to contextual function (implementation protocol)

Prepared 2026-10-22 after inspecting e07 results. This is an exploratory
follow-up, not a completed experiment or a new e05/e06 criterion.
No e08 model run has started. Finish the existing e05b/e06 work first; use
only the saved artifacts and local VRAMorama resources, with no new training,
rentals or model downloads. Freeze remaining choices below before examining
e08 intervention results.

## Question and scope

Do different contextual families depend on separable signed contributions
from components contacting the same neuron? The strongest proposed evidence
is a held-out double dissociation: component group A has a larger effect in
family A than B, while group B has a larger effect in family B than A.
One group affecting two families establishes shared causal involvement but
does not by itself establish separability or interference between features.

Fix the e07 candidates: layer-0 gate coordinate 196, layer-1 gate coordinate
33, and layer-1 up coordinate 1037, in both decomposition seeds. These are
three selected cases from one target Qwen3-0.6B model, not a representative
sample of neurons or independent language models. Component IDs are local
to each decomposition; do not equate IDs between seeds.

For each candidate, retain all 768 rank-one subcomponents when collecting
activity. Mark the e07 top-eight contact set for comparison rather than
discarding the other contributions. Preserve the decomposition's delta
term and the other model modules throughout interventions.

## Context collection and semantic labels

Use the cached 128-token rows and fixed seed-0 row permutation from
`e04_common.heldout_part`. Within its causal range, reserve the first 512
rows for discovery and the next 512 for evaluation, identical across seeds.
Never wrap or silently substitute rows if the cache is shorter. These ranges
are disjoint within e08, but reuse the corpus and may overlap e05 causal
evaluation: this is not a fresh external confirmatory dataset. Do not use
e08 labels or results to revise e05/e06.

At each token, record row/token identity, position, candidate activation,
CI, and signed component contributions at the relevant gate/up projection.
The existing dataset-attribution aggregates do not replace this token-level
record. Save the surrounding prefix and next token; avoid including future
tokens in the information used to explain a next-token prediction.

Select at most one candidate token per row, by largest absolute activation,
excluding position 0 from the primary analysis. For discovery, inspect up to
128 highest-activation rows per candidate, breaking ties by row/token ID.
Report position 0 separately, because the prior study identified a strong
position effect. Log how this activation-based sampling limits generality.

Assign contextual families without using component IDs, contributions or
intervention outcomes. A cached local text encoder may propose clusters;
record its identity, normalization, algorithm and seed before use. If no
encoder is cached, use blinded manual context labels; lexical clustering
alone is not evidence of semantic similarity. Do not download an encoder.
The annotation rule, family descriptions, ambiguous/other policy and frozen
classifier or assignment rule must be recorded before evaluating held-out
effects. Two annotators and agreement reporting are preferred; if only one
is available, disclose it rather than inventing agreement.

Carry forward at most two nonredundant families per candidate, ranked by
discovery frequency with a fixed label-name tie break. Require at least
32 discovery rows and 32 evaluation rows per family. If fewer than two
families qualify, report the case as insufficient evidence; do not search
additional neurons or change the row budget to obtain a positive result.
Evaluation family assignment and token selection must not depend on
intervention effects.

## Signed contribution and intervention definition

For a projection output coordinate j, record

    z_c(t,j) = CI_c(t) * activation_c(t) * U_cj.

Use the actual signed product, not the sign of an isolated factor: a
rank-one factor can flip signs without changing its contribution. Also
record raw, unmasked contributions and full-versus-CI-masked reconstruction
error. Gate/up preactivations combine nonlinearly downstream; z_c is not
itself a signed final-logit effect or a semantic label.

Using discovery rows only, rank contacting components by their difference
in mean signed contribution between the two families. Take at most eight
components at either end as disjoint groups A and B; publish the complete
ranked table, group sizes and retained contribution mass. Selection is
exploratory and must be checked out of sample.

Intervene on a group's full projection-output contribution at the selected
token, rather than changing a factor's sign in the saved checkpoint. Let
mu_G be its discovery mean in the same position class and set

    z_G' = mu_G + alpha * (z_G - mu_G), alpha in {0, 1, 2}.

This gives opposite signed perturbations around the observed contribution,
with alpha=1 as a numerical identity check. Hold the baseline CI masks
fixed across paired interventions; distinguish this controlled intervention
from a separate, optional experiment that recomputes CI after intervention.
Do not overwrite either checkpoint. Include a coordinate-only removal of
the same group's contribution at j to test whether effects specifically
depend on the shared coordinate; the full-vector effect alone cannot show
that neuron j mediates the effect.

Measure paired changes in observed next-token log probability, full-
vocabulary KL to the unmodified CI-masked baseline, and prespecified
family-specific token scores where a defensible score can be defined before
evaluation. Log probability alone measures prediction disruption, not the
meaning of a feature. Report reconstruction error against the original
target alongside these effects. Failed e05 quality gates do not become e08
passes: interpret any such decomposition's effects with that limitation.

Controls: identity hook; same-size random groups matched on discovery
firing-rate decile and contribution-energy decile; family-label permutation
at the row level; matching or stratifying family comparisons on activation
magnitude and position. Use 100 random groups and 1,000 row-level label
permutations, RNG seed 81022. If exact matching is impossible, report the
case and mismatch; do not silently relax bins. Resample rows, not correlated
tokens, for 1,000 bootstrap intervals.

Report the complete group-by-family effect matrix and both directions of
the interaction. Treat the six candidate-by-seed cases as exploratory;
report permutation p-values with Holm adjustment across the six primary
double-dissociation tests, with effect sizes and row counts. No universal
polysemanticity claim follows from passing selected examples. Describe an
effect as interference only if a separately defined feature-specific
tradeoff is demonstrated; global KL changes do not establish interference.

## Companion: energy-weighted percolation

Use every nonzero coordinate, with q_cj = a_cj^2 / sum_j(a_cj^2), for the
same 12 maps. This retains direction energy rather than only the largest
eight entries, but still weights directions equally and ignores activity,
component scale and delta. It remains a geometric diagnostic.

Define the stochastic graph explicitly. Independently retain contact cj at
exposure t with probability

    r_cj(t) = 1 - exp(-t * N * q_cj), N = 3072.

Use t=0 plus 40 logarithmically spaced points from 1e-5 through 4
and 32 nested draws, base RNG seed 81023. Zero weights never
create edges. Equal-energy coordinates have retention 1-exp(-t), making
exposure interpretable; t is not e07's uniform bond probability p. Report
both edge fraction and expected retained energy versus t, including any
failure to reach a 50% component cluster on the fixed grid. Do not expand
the grid after seeing curves. Also report the number of contacts needed
per direction to retain 50% and 90% energy, without conflating energy
thresholding with independent bond percolation.

Controls: uniform q=1/N and independent within-row coordinate permutations,
which exactly preserve each row's energy spectrum but disrupt shared
coordinate preferences. These controls do not preserve coordinate strengths;
label that limitation. A strength-preserving control requires a separately
specified valid nonnegative weighted null and mixing check before use.
Do not call a binary degree-preserving rewire strength-preserving.

Use identical component denominators and report full curves, crossover
differences and integrated differences descriptively. The e07 formula
p^2*kappa_C*kappa_N assumes uniform bond retention; do not substitute an
average weighted retention probability into it as an exact prediction.
A weighted branching prediction needs its own multitype or nonbacktracking
derivation and independent calibration. This companion can establish
robustness to the contact definition; it cannot supply semantic labels.

## Execution and outputs

The implementation is complete and tested on synthetic cases; artifact
execution is deferred until e05b/e06 has finished and its results have been
reviewed and committed. The following execution choices are frozen before
e08 measurements. Runtime checks verify reconstruction, identity hooks and
agreement between saved and recomputed contextual contributions. Review available disk/RAM and acquire VRAMorama
leases only after the existing e05b/e06 pipeline finishes. Never reduce
its batch counts or alter its guards to fit e08.

Proposed outputs: a row-index split manifest, token-context table, label
rules/assignments, signed-contribution summaries, paired intervention
effects and all controls, weighted percolation curves, and an e08 results
report distinguishing geometry, contextual association, causal selectivity
and interference. Commit completed local work; never push without an
explicit request. Independent target models remain a later generalization
test and require separately authorized artifacts or compute.

## Frozen implementation choices, before any e08 model run

The human authorized implementation and deferred local execution on
2026-10-22. No e08 artifact measurements preceded the choices here.

- **Context representation:** reuse the cached original Qwen3-0.6B model's
  input to lm_head at the selected token, normalized to unit length. This
  is a contextual representation, not a separately validated semantic
  encoder. No model download is allowed. Four spherical k-means++ clusters
  (200 iterations, RNG 81024 + 3*seed + candidate index) are proposed from
  the top 128 discovery rows only; evaluation uses their frozen centroids.
  Degenerate or nonconvergent clusters are left unscored.
- **Blinded review:** a sparse follow-up reviews 16 discovery prefixes per
  cluster (central and dispersed examples), without contributions, group
  selection, next-token targets or intervention effects. A cluster needs
  an explicit contextual-family description, rationale and at least 12
  supporting reviewed prefixes to be accepted. Use an ambiguous/other
  label and coherent=false when uncertain or when similarity is merely
  lexical formatting. Record reviewer identity; a single AI review is not
  independent human annotation or an agreement study. Labels are frozen
  in out/e08/labels.json before any effect is evaluated. Distinct family
  labels require distinct contextual meanings, not just different names.
- **Token eligibility/storage:** positions 1-126 can have a primary next-token
  score within the same 128-token row; position 127 has no within-row target.
  Record full component activation/CI vectors at each primary token and at
  position zero, plus all-token discovery firing/energy/position-class sums.
  Do not materialize redundant multi-GB all-token component tensors. All
  row IDs, prefixes and checkpoint hashes are retained for reconstruction.
- **Delta and hooks:** unlike the e05 gate, e08 retains the fixed full-rank
  residual W_target - W_components in every decomposed module, in baseline
  and interventions. CI=1 plus delta must reconstruct original logits
  (atol .003, rtol .001). The existing e05 gate is neither recomputed nor
  relabeled. Additive hooks act on a component module's output after
  replacement, at the selected row/token, before the MLP nonlinearity;
  downstream modules respond normally with their baseline CI masks fixed.
  The identity-hook maximum logit error must be <=.003. Saved/recomputed
  selected coefficients must agree at atol 1e-4, rtol 1e-3.
- **Groups/matching:** rank only the e07 top-eight contact set at the candidate
  coordinate; group size is min(8,floor(number of contacts/2)). Match random
  controls from all 768 components excluding both selected groups, on exact
  discovery firing and full-vector contribution-energy deciles. Keep tied
  values in the same bin. Draw without replacement within a group; the
  same component may appear in different control draws. A missing exact
  match leaves that case unscored; no bin relaxation.
- **Primary effect:** the smaller of the two directed family-by-group KL
  interactions for alpha=0 full-vector interventions. Both must be positive;
  report one-sided permutation p and Holm adjustment across all six fixed
  cases, including p=1 for unscored cases. This measures selective contextual
  disruption. Observed-next-token log-probability change is secondary. No
  independent feature-specific token contrast is defined, so e08 cannot
  establish feature interference from these scores alone. Alpha=2 effects,
  coordinate-only alpha=0 effects, and all matched controls are reported.
- **Strata/resampling:** four fixed position bands (position // 32) crossed
  with discovery absolute-activation quintiles. Keep common cells with >=2
  rows per family; weight cells by their smaller family count. Require >=32
  raw evaluation rows and >=16 retained matched rows per family. Permute
  row labels and bootstrap rows within these cells, 1,000 times each.
- **Weighted-grid correction:** the draft linear grid began at t=.1. Before
  running, the uniform dense-graph degree calculation showed that mean
  degrees scale as N*t and C*t at small t, giving a branching scale near
  1/sqrt(C*N), about 0.00065 here. A fixed logarithmic 41-point grid resolves
  that earlier regime without choosing points from measured curves. No
  measured result motivated this correction. Null row permutations are
  drawn independently for each of the 32 draws. Map RNGs are 81023 +
  100*seed + 10*layer + map index in seed/layer/gate-up-down order.
- **Resources:** native driver waits without model calls, checking local
  prerequisites every five minutes and logging at most hourly. It requires
  the original workers to exit, successful e06 outputs, docs/e06_results.md,
  and results/e08/prerequisites_reviewed.json naming the local e06 commit.
  It never invokes training or cloud tools. CPU stages have 64 MiB VRAMorama
  bookkeeping leases; weighted geometry has a 768 MiB Windows commit cap.
  Context collection uses 6 GiB and paired effects 8 GiB leases; one context
  row at a time, four intervention variants per forward batch. This batching
  changes no row counts, controls or resamples. At least 1.5 GiB free disk is
  required. Stages fail visibly rather than relax guards or settings.
  An OS-held driver lock prevents duplicates. Atomic per-row outputs allow
  reviewed recovery without repeated completed rows. Protocol/code hashes
  must match on resume; label/plan hashes bind paired effects to their inputs.

Runner: experiments/e08_pipeline.py --wait-for-e05b-e06, launched detached
through a small VRAMorama lease. It pauses for blinded review after context
collection if labels are absent. The same sparse four-hour heartbeat handles
that review and resumes it; no ten-minute agent monitor is introduced.
The design is implemented; no e08 results are claimed until files and exit
statuses are inspected. Reports are generated only after all six cases are
accounted for, including unsupported cases.

## Implementation validation

The local suite passes 83 tests, including independent weighted-graph
connectivity calculations, exact reduction to uniform Bernoulli percolation,
discovery/evaluation separation, exact-bin matching, positive and negative
double-dissociation examples, multiplicity correction, blinded-review gates,
retained-delta reconstruction and token-local signed hooks on synthetic
projections. All e08 modules compile. These are CPU implementation checks;
real artifact reconstruction and batching checks run after e05b/e06 finishes.
No real e08 outcomes are claimed by these tests.

## Launch setup

Create out/e08 before submitting a detached VRAMorama command whose --out
argument points into that folder. VRAMorama opens its output file before
the Python driver can create directories. The first submission failed at
this file-open step, before running Python or measuring any e08 artifact;
its lease/worker absence was checked before one corrected launch. The
corrected native driver is admitted and waiting for the predecessor review.
No protocol criterion, sample count or guard changed during this correction.
