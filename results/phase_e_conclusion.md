# Phase E Conclusion: RQ5 (Robustness/Collapse)

**Pre-registration:** written and pushed BEFORE any Phase E experiment code existed
(commit `92c800c`), adapting the original hypothesis to what Phases C/D actually
established (no clean baseline ablation signature exists on confounded topology; genuine
competence is specifically on chain/fork/collider). See `PREREGISTRATION.md` for exact
predictions and the corrected 4-condition design (fixing the starter spec's Conditions A
and D being mechanistically identical as written).

**Scale actually run:** starting from the iteration-135 checkpoint, 3 generations for
Conditions A, B, C; Condition D reached generation 2 before the Colab session's compute
budget ran out (generation 3 was not completed). This is disclosed as an incompleteness,
not glossed over -- see limitations below.

## Headline result: the opposite of the pre-registered prediction

**Conditions A, B, and C (all three recursive-training variants) show *zero* measurable
degradation in genuine causal competence across all 4 generations (0, 1, 2, 3).** The
discrimination-test correlation (Phase D's method: predicted-vs-true answer correlation on
chain/fork/collider, r should be near 1.0 for genuine competence) is **identical to more
than 10 decimal places** across every checkpoint in these three conditions:
chain=0.9872364313401436, fork=0.98452822480056, collider=0.9988943723905196 -- the same
numbers at generation 0 (the untouched Phase A checkpoint) as at generation 3 (three rounds
of recursive self-training later), for all three conditions.

**Condition D -- the non-recursive control, expected to be the STABLE baseline -- is the
one that collapsed.** By generation 1, fork and collider's correlation became undefined
(the model's predictions had zero variance -- a telltale sign of constant output). By
generation 2, chain's correlation flipped to **-0.987** (the model was answering
*backwards*), and the ablation check produced **zero parseable outputs at all** (complete
answer-format breakdown). A direct diagnostic (querying two very different questions and
checking the model's argmax next-token) confirmed this is genuine: D-generation-2 answers
`" 10%"` to *every* question regardless of content, while A-generation-3 correctly answers
`" 80%"` and `" 10%"` to the same two questions respectively -- real, input-sensitive,
preserved discrimination in the recursive condition, and genuine collapse in the
"stable" control.

## Why this rules out the two obvious alternative explanations

1. **Not a caching/evaluation bug.** `load_checkpoint_model` constructs a fresh model
   instance and loads new weights from disk on every call -- there is no code path that
   could return stale results. The identical correlations across A/B/C's four
   *different, actually-trained* checkpoints were independently confirmed by direct
   token-level inspection, not just the aggregate correlation number.
2. **Not a coincidental collapse to a lucky constant.** A constant-output model
   mathematically cannot produce a correlation near 1.0 with genuinely varying true
   answers (a constant has zero variance, making Pearson correlation undefined, not high).
   The r=0.987-0.999 values in A/B/C require real, preserved input-sensitivity -- confirmed
   directly by the diagnostic showing different answers to different questions.

## Pre-registered predictions: formally not supported, but for an informative reason

- **Prediction 1** (Condition A's discrimination correlation degrades faster than
  Condition D's): **not supported** -- Condition A showed *no* degradation at all, while
  Condition D showed *catastrophic* degradation. The prediction's directional assumption
  (recursion is more damaging) is falsified in the opposite direction from what was
  predicted.
- **Prediction 2** (Condition B degrades less than Condition A): **vacuously
  not-meaningfully-testable** -- neither condition degraded at all, so there's no
  degradation gap for real-data anchoring to narrow. B's correlations exactly match A's
  and C's throughout.
- **Prediction 3** (output diversity shrinks in A, less so in B/D): **inconclusive** -- the
  diversity metric (std. dev. across 10 repeated samples of one fixed prompt) was noisy and
  mostly near-zero even at generation 0 across all conditions (a limitation of this
  particular metric's design, discussed below), and showed no clear trend distinguishing
  conditions. Not a meaningful test as implemented.

**Using the interpretation table from PREREGISTRATION.md, this is closest to a novel
variant of "Pattern 4" (training beyond Phase A drifts the model) -- except the drift is
concentrated specifically in Condition D's training regime, not a general property of
"further training."**

## The likely mechanistic explanation: an optimization-dynamics confound, disclosed

Condition D's implementation differs from A/B/C not just in "fresh vs. self-generated
data" (the intended manipulation) but also in **optimization dynamics**: D performs many
small-batch (~12-question), immediate-gradient-update iterations in sequence (43+ separate
fine-tuning calls per generation, each updating weights right away on a tiny batch),
whereas A/B/C generate one larger fixed corpus (~90-120 accepted pairs) once per generation
and then run 3 clean epochs over that fixed corpus. This is a real, uncontrolled second
difference between the conditions, and the most likely candidate explanation for D's
collapse: many repeated small-batch gradient updates at this learning rate (1e-5) may be
substantially more prone to compounding instability / catastrophic forgetting than fewer,
larger, cleaner update steps -- independent of whether the training data is self-generated
or fresh. **This means Phase E's actual, best-supported finding is narrower than "recursion
is safe" -- it is "in this comparison, the specific non-recursive training regime tested
was less stable than the specific recursive regime tested, for reasons that may be more
about optimization dynamics (batch size / update frequency) than about recursion per se."**
A cleaner follow-up would match A/B/C and D's optimization dynamics exactly (same batch
size, same epoch structure) while varying only the data source, to isolate recursion from
this confound.

## The confounded-topology floor: unchanged everywhere

Confounded-topology accuracy remained exactly 0.0% in every single (condition, generation)
cell evaluated -- the Phase C/D floor is neither worsened nor "fixed" by any of the four
training regimes tested, including the one that collapsed catastrophically on the
*unconfounded* topologies. This is a clean, if unsurprising, replication: there was no
competence there to begin with, and Phase E's manipulations don't change that fact either
way.

## Ablation-collapse pattern (not pre-registered as pass/fail, tracked qualitatively)

Phase C's original finding -- that mean-ablating the candidate circuit collapses the
model's confounded-topology output to a single constant -- persisted in most cells (a
single distinct value across 20 test questions), though the specific constant drifted
across generations and conditions (0.1 -> 0.8 -> 0.5 -> 0.95 in various cells, never
settling on one fixed value). Condition D generation 2 is the exception: ablation there
produced **zero** parseable outputs at all, consistent with that checkpoint's general
collapse. The qualitative "ablation causes collapse, not directional correction" pattern
from Phase C held up in every condition where the base model itself hadn't already
collapsed.

## Limitations

1. **Condition D generation 3 was never run** (Colab compute/session limits after an
   unrelated runtime reset lost an earlier full attempt, and by generation 2 the model had
   already collapsed to the point where a generation-3 result would likely be
   uninformative). The trend from gen 0->2 is clear enough to draw the headline conclusion,
   but the full 4-generation comparison is incomplete for D specifically.
2. **The output diversity metric was underpowered** -- a single fixed test prompt, 10
   repeated samples, at generation 0 already showing near-zero variance in most conditions.
   This metric did not provide useful signal as implemented and would need redesigning
   (more prompts, more samples, or a different diversity measure like distinct-n-grams
   across a larger generation set) to test Prediction 3 meaningfully.
3. **The optimization-dynamics confound** (batch size / update frequency differing between
   D and A/B/C) means this result cannot cleanly isolate "recursion" as the causal factor
   in either direction -- it shows something real happened, and something asymmetric
   between the conditions, but not unambiguously that the asymmetry is *about* recursion
   specifically rather than the accompanying difference in training regime.
4. Only 3-4 generations were tested, starting from one checkpoint (iteration 135). Whether
   A/B/C's apparent perfect stability would hold over many more generations, or whether
   D's specific collapse pattern is reproducible with a different seed, is untested.

## Honest bottom line

Phase E did not find evidence that recursive self-training specifically damages the causal
reasoning circuit faster than raw competence, as pre-registered. It found something
different and, arguably, more surprising: **across the conditions actually tested, the
non-recursive continued-training control was the fragile one**, collapsing to
context-insensitive constant outputs within 2 generations, while all three recursive
variants (including the one with no real-data anchoring at all) showed no measurable
degradation whatsoever in genuine, verified causal discrimination ability. Given the
disclosed optimization-dynamics confound, the honest claim is narrower than "recursion is
protective" -- it is that *this specific* non-recursive training regime was unstable in a
way *this specific* recursive regime was not, and disentangling why (data source vs.
update dynamics) is the natural next step, not yet answered here.
