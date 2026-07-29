# Phase D Conclusion: RQ4 (Behavioral Correlate)

**Setup:** matched implicit/explicit intervention prompt pairs (1 implicit + 3 explicit
do-calculus-scaffolding variants per question), built from each topology's actual
Phase-A-trained (treat_var, target_var) pair (a lesson carried over from Phase C: using
untrained variable pairs would confound the result with out-of-distribution incompetence).
n=15 questions/topology, evaluated at all 10 available Phase A checkpoints (0, 15, 30, ...,
135 -- Phase A found no single clean transition, so per Step 3.2's alternative guidance,
every saved checkpoint was evaluated rather than concentrating density around one
hypothesized moment). Best-of scoring across explicit variants (a pair counts as
explicit-correct if ANY variant is correct). Tolerance 0.10, matching Phases A-C.

**Disclosed confound:** explicit prompts average 245 characters vs. implicit's 137 (~1.8x
longer) -- length was not controlled for (no filler-padding was added to the implicit
version), so any observed explicit-over-implicit advantage cannot be fully separated from
a generic "longer prompt" effect. This should be read as a limitation of what follows, not
a fixable-after-the-fact issue -- the templates were fixed before evaluation began.

## Step 2.4 checkpoint (prompt design smell test)

Passed: manually reading 6 random pairs confirmed matched topology/question/true-answer
between implicit and explicit versions, and confirmed the explicit variants add
do-calculus-specific content (naming the intervention, do-operator notation, "ignore normal
causes") rather than generic elaboration.

## Results

**Per-checkpoint, per-topology gap (explicit accuracy minus implicit accuracy), with
Wilson-score 95% CIs (n=15, so a normal approximation would be unreliable near 0/1):**
full table in `src/phase_d_analysis.py` output; plot in `results/phase_d_gap_plot.png`.

**Once CIs are accounted for, almost none of the apparent gaps are statistically
meaningful.** Across all 40 (checkpoint x topology) cells, only **one** — chain at
iteration 90 (implicit 0.53 [0.30-0.75] vs. explicit 0.00 [0.00-0.20], non-overlapping CIs)
— reaches the threshold this project used throughout (Wilson CI non-overlap) for a real
effect. Every other apparent gap, including some as large as 33-53 percentage points in
raw terms (fork at iteration 75: -0.33; chain at iteration 0 and 15: ±0.20), has overlapping
confidence intervals given the small n=15 sample and is not distinguishable from sampling
noise.

**The one statistically real effect found is a large NEGATIVE gap, not the hypothesized
positive one.** At iteration 90, explicit scaffolding drove chain accuracy from 53% down to
0% -- scaffolding actively hurt, rather than helped. This is the specific "possible go
wrong" scenario flagged in Step 8 ("scaffolding may confuse the model at some
checkpoints"), observed directly rather than merely anticipated.

**Confounded topology -- the one topology where a real gap would actually be diagnostic of
intervention-vs-association reasoning (see Phase C) -- shows EXACTLY 0% accuracy for both
implicit and explicit prompts at all 10 checkpoints, with zero variance.** Not even the
most heavily scaffolded variant (explicit variant 3, which writes out
`P(target | do(var=val))` notation directly) ever rescued a single confounded-topology
question, at any point across the entire training run.

## Judgment, using Step 6.2's framing options

This is closest to **"the gap was never large to begin with, but for a specific and
important reason that the framework anticipated"**: on chain/fork/collider, apparent gaps
are noise (CIs overlap); on confounded, there is no gap because there is a **floor, not
parity** -- both prompt styles are at 0%, not both at some shared high competence level.
Per Step 6.2's own guidance to "rule out (b) [insufficiently distinct scaffolding] before
endorsing (a) [native handling]": the Step 2.4 smell test already argues against (b) for
prompt content, but the length confound (disclosed above) is a real, uncontrolled
alternative explanation for the (few, non-significant) apparent explicit advantages seen on
chain at iteration 0. Neither (a) nor (b) is a good fit for confounded specifically --
option (c), not listed in the original framing but the one this project's Phase C already
established: **the model has no real underlying competence for confounded-topology
intervention questions to express, in either prompt style, at any point in training.**

## Cross-phase integration (Step 7)

See `results/cross_phase_integration.png` -- three stacked panels (Phase A per-rung
accuracy, Phase A LLC, Phase D gap-by-topology) on a shared training-iteration axis.

**This provides independent, convergent corroboration of Phase C's central finding, via a
completely different method.** Phase C ablated the internal circuit and found the
un-ablated model was already at 0% interventional accuracy on confounded-topology
questions. Phase D never touches the model's internals at all -- it only varies the
prompt -- and finds the same 0% floor, at every single training checkpoint, regardless of
how much explicit causal-reasoning scaffolding is provided. Two independent probes (one
mechanistic, one behavioral) agree on the same limitation. This is the "two independent
thermometers" logic the phase's own introduction described -- just converging on a
**negative** finding (a genuine incapacity) rather than the hoped-for positive one (a
capacity that internalizes over training).

**No LLC/gap alignment could be meaningfully tested for confounded**, since the gap is
identically zero throughout (there's nothing for an LLC transition to align with). For
chain/fork/collider, the LLC's single early transition (iterations 0-30) does not
correspond to any statistically real gap-closing event in those topologies either -- but
since those topologies were never diagnostic of intervention-specific reasoning in the
first place (per Phase A/C's design: no real association/intervention divergence), this
null result is expected rather than informative.

## Honest interpretation

Phase D adds a fourth line of evidence, and it points the same direction as Phase C: this
model's apparent training-time "interventional competence" (as pooled-topology accuracy in
Phase A, or as a transferring circuit in Phase B) does not include genuine competence on
the one topology that actually requires distinguishing intervention from association. No
amount of prompt-level scaffolding surfaces competence that Phase C already showed isn't
mechanistically present. This substantially narrows what Phases A/B's more optimistic
readings can claim: whatever the model learned, it did not include real do-calculus
reasoning for confounded structures, and this is now supported by two independent methods,
not one.

## What would strengthen this

1. **Control for the length confound directly** -- either pad implicit prompts with neutral
   filler to match explicit length, or run a length-matched-but-content-free "explicit"
   control (same length, no causal-reasoning content) to isolate content from length.
2. **Investigate the one real effect (chain, iteration 90) further** -- is this a
   genuine, reproducible "scaffolding actively confuses the model at certain training
   points" phenomenon, or a one-off artifact of that specific checkpoint's idiosyncrasies?
   A replication run (different seed) would help distinguish these, following the same
   cross-environment-replication practice used successfully in Phase A.
3. **Larger n per topology** -- n=15 gives wide Wilson intervals; doubling or tripling it
   would sharpen the ability to detect real, smaller gaps if they exist.

## Status for Phase E

Phase E (recursive training) re-uses Phase C's ablation-signature procedure. Phase D's
findings don't change what Phase E should test, but they reinforce the same caution Phase
C raised: whatever gets tracked across recursive generations should be checked for whether
it reflects genuine capability or a pre-existing shortcut, on confounded-topology
questions specifically, given both Phase C and Phase D now agree that shortcut is
already the model's total behavior there.
