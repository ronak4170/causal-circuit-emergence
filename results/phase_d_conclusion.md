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

## Follow-up 1: length confound, resolved (`src/phase_d_evaluate_control.py`)

A third condition was added: `neutral_control` -- the implicit question padded with
content-free filler text to match each pair's own mean explicit-variant length (246 vs.
245 characters, essentially exact). Re-run across all 10 checkpoints.

**Result: length never mattered, anywhere.** `neutral_control` accuracy was **identical**
to `implicit` accuracy at every single one of the 10 checkpoints, for every topology,
including at the two checkpoints where `explicit` had previously diverged (fork at
iteration 75: implicit/neutral_control both 0.67, explicit 0.33; chain at iteration 90:
implicit/neutral_control both 0.53, explicit 0.00). This cleanly attributes both of those
effects to the scaffolding's *causal-reasoning content* specifically, not to the extra
length -- ruling out the length confound as an alternative explanation for the one
statistically real effect this phase found. Full results:
`results/phase_d_control_results.pkl`.

## Follow-up 2: does chain/fork/collider's accuracy reflect genuine reasoning?
(`src/phase_d_discrimination_test.py`)

Since association and intervention answers coincide for these topologies by design, high
tolerance-accuracy alone couldn't rule out "the model outputs a lucky/memorized constant."
Tested directly at checkpoint 135 (n=40/topology, fresh questions varying `do_value`):
correlation between the model's predicted probability and the true answer, and whether the
model's predictions correctly shift direction between `do_value=True` and `do_value=False`
groups (a true constant-output model would fail this even if individually "lucky" answers
landed within tolerance).

**Result: strong, genuine discrimination.** Correlation was 0.985 (chain), 0.989 (fork),
0.998 (collider) -- and in all three, the model's mean prediction correctly shifted in the
same direction as the true answer between `do_value` groups (e.g. chain: predicted 0.85 vs.
0.10 for True vs. False, matching the true answer's own 0.85-ish vs. 0.10-ish split). See
`results/phase_d_discrimination_scatter.png`. **This rules out the "memorized constant"
concern**: the model genuinely tracks which direction the intervention was applied, for
unconfounded topologies.

**One caveat worth flagging precisely:** the model's predictions are not continuous --
each topology shows exactly two clustered output values (e.g. chain: ~0.85 or ~0.10), not
graded probability estimates matching the true answer's finer variation within each
`do_value` group. The high correlation partly reflects that the true answers are also
fairly tightly clustered within each group, so a coarse binary-ish classifier can achieve
high linear correlation without doing graded probability estimation. The honest
characterization: the model has learned a genuine, input-sensitive **binary discrimination**
of the intervention's direction for unconfounded topologies -- real competence, but coarser
than full continuous causal inference.

## Revised overall picture

Combining both follow-ups with the original Phase D result: **the model has real,
verified, direction-correct (if coarse) causal competence specifically on the unconfounded
topologies (chain, fork, collider) it was trained on -- and a complete, floor-level absence
of that same competence on the one topology (confounded) that actually requires
distinguishing intervention from association.** This is a sharper, more informative
picture than the original Phase D write-up could support alone: it is not that the model
"has no causal competence" in general (Follow-up 2 rules that out for the unconfounded
case), and it is not that Phase D's apparent effects were artifacts of prompt length
(Follow-up 1 rules that out too) -- it is specifically that whatever mechanism produces
correct unconfounded-topology answers does not extend to the confounded case, exactly
where Pearl's do-calculus distinction actually bites.

## Status for Phase E

Phase E (recursive training) re-uses Phase C's ablation-signature procedure. The revised
picture from the two follow-ups sharpens what Phase E should track: there are now two
verified, distinct baselines going in -- genuine (if coarse, binary-ish) causal
discrimination on chain/fork/collider, and a complete floor on confounded -- rather than
one uniform "shortcut everywhere" story. Phase E should track whether recursive training
degrades the *genuine* unconfounded competence (a real capability that could be lost) at a
different rate than it affects the *already-absent* confounded competence (which has
nothing left to lose), rather than treating "interventional competence" as a single
undifferentiated thing across topologies.
