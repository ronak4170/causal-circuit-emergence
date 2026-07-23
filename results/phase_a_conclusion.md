# Phase A Conclusion: RQ1 (Emergence)

**Run:** GPT-2 small, warm-started on association-only SFT (400 examples, 3
epochs), then RFT across a curriculum of chain/fork/collider/confounded
topologies x all three Pearl rungs, 150 iterations, 4 questions/rung/iteration,
4 samples/question, tolerance 0.10. Local CPU run (`results/phase_a_log.pkl`,
`results/phase_a_checkpoints/`). A second, independent Colab GPU run of the
same pipeline (different seed/hardware) produced a qualitatively matching LLC
trend (see the LLC section below) but is not used for the accuracy panel
since its checkpoints were lost to a Colab runtime reset before an LLC pass
could be matched to that specific accuracy log.

Artifact: `results/phase_a_plot.png` (two panels: smoothed per-rung accuracy
with detected-jump markers, and LLC vs. iteration).

## Jump detection criterion (stated before reading conclusions off the plot)

Rolling mean (window = 10 iterations, no zero-padding at the boundary) over
each rung's raw per-iteration accuracy; a "jump" is the first iteration where
the smoothed curve exceeds its value 10 iterations earlier by >20 percentage
points AND the remaining training tail stays elevated (not a transient blip).
See `src/phase_a_plot.py` for the exact implementation.

**Honesty note on blinding:** this criterion was written and coded before
running the detection script, but *not* before ever looking at the training
data — the per-iteration accuracy numbers were watched live, iteration by
iteration, throughout the ~2.5 hour training run (visible in this
conversation's history) before this script was written. This is weaker than a
true pre-registration and is disclosed here rather than glossed over. The
detection thresholds (20pp, 10-iteration window) were chosen based on general
judgment about what would count as a meaningful transition, not by tuning
against the observed curves to produce a particular answer.

## What the data show

**Detected jumps:** association: none; intervention: iteration 10;
counterfactual: iteration 86.

**Association never shows a detectable jump because it starts at ceiling.**
The required warm-up step (Step 2.3) trained the model on association-only
examples before RFT began, and it worked exactly as intended -- association
accuracy opens at ~100% and fluctuates in a 0.7-0.95 band for the rest of
training. This means **RQ1's "association emerges first" claim is untestable
in this run design**: there was no room left for association to "emerge"
during RFT, since the warm-up had already saturated it. This is a design
confound, not evidence against Pearl's predicted order -- a cleaner test
would either skip the association-specific warm-up (at the cost of the
reward-sparsity problem it was added to solve) or use a warm-up so brief that
association starts below ceiling.

**Intervention's "jump" at iteration 10 is early and modest, not a clean
discrete transition.** After the initial rise (visible in the plot as the
climb from ~0.3-0.6 in the first 10 iterations to a ~0.5-0.75 band by
iteration 20-30), intervention accuracy spends the rest of training
oscillating noisily between roughly 0.4 and 0.7 with no further structural
change -- it does not look like "one clear phase transition" so much as
"settling into a noisy plateau early."

**Counterfactual's jump at iteration 86 is the closest thing to an
interesting late-stage signal**, rising from a trough around iteration 75-80
(as low as 0.15) to a sustained ~0.4-0.65 band afterward -- but even this is
not clean: it dips again around iteration 105-120 before a final rise near
the end of training. Calling this a genuine "phase transition" rather than
noise around a slow upward drift would be overclaiming given the sample
sizes involved (4 questions/rung/iteration means each raw point is quantized
to {0, .25, .5, .75, 1.0}).

**LLC shows one clear transition, in the first ~15 iterations** (15.4 -> ~17.7,
see `results/phase_a_llc.pkl`), then plateaus in the 17.1-17.9 range with
overlapping error bars for the rest of training -- no second transition
appears anywhere later, including at counterfactual's iteration-86 jump. The
early LLC rise's timing loosely overlaps intervention's early rise (both
happen in the iteration 10-15 window) but this is weak evidence at best given
how early and confound-prone that window is (it's also where the model is
adapting away from the pure-association warm-up distribution to the full
mixed-rung curriculum, which would itself be expected to raise model
complexity regardless of any rung-specific competence change). **A second,
independently-trained Colab run reproduced the same LLC shape** (an early
jump within the first 15-30 iterations from ~16.5 to ~19, followed by a
plateau/mild-decline for the remainder of training) -- this replication
increases confidence that the LLC's *early-jump-then-plateau* shape is a real
property of this training setup, not a fluke of one run's noise, even though
it does not confirm any specific *rung-linked* transition.

## Judgment: Partial support, leaning toward falsification

Using the pre-specified categories:

- **Full support** would require all three curves to show distinct,
  well-separated jumps, LLC changes aligned with each, and the order
  association -> intervention -> counterfactual. **Not observed.**
- **Partial support** allows jumps detected but imperfectly ordered, LLC
  aligning with some but not all, or one rung simply not emerging cleanly on
  a model this small. This is the closer fit: two jumps were detected
  (intervention, counterfactual), one rung was structurally untestable
  (association, due to the warm-up ceiling), and LLC aligns weakly with at
  most one of the two detected jumps.
- **Falsification** requires smooth accuracy rise with no detectable jumps
  and no LLC signal anywhere. Not quite this either -- LLC *does* show one
  clear, replicated transition, and two rungs *do* show detected
  (if noisy/unclean) jumps.

**Overall: partial support, weak.** The strongest honest claim this run
supports is: *"in this small-model, small-batch RL post-training run, we
find one clear, replicated LLC phase transition early in training, and
noisy, only loosely jump-like accuracy dynamics in the intervention and
counterfactual rungs that do not cleanly align with each other or with the
LLC transition; the association rung's predicted early emergence could not
be tested because a required warm-up step had already saturated it before
RFT began."* This is a legitimate, reportable result for a project at this
scale, but it should not be written up as confirming Pearl's staged-emergence
hypothesis -- the evidence is too noisy and too confounded by the small
per-iteration sample size (4 questions/rung) and the warm-up design choice to
support that stronger claim.

## What would make this a cleaner test next time

1. Larger per-iteration batches (more questions/rung) to reduce the
   accuracy-quantization noise that dominates the raw curves.
2. Either skip or drastically shorten the association-only warm-up so
   association starts below ceiling and its emergence (or lack thereof) is
   actually observable.
3. More LLC checkpoints, especially densely sampled around iteration 10-30
   and 75-90, to determine whether the intervention/counterfactual jumps have
   real corresponding LLC structure that the current 15-iteration checkpoint
   spacing is too coarse to resolve.
4. A GPU run at larger scale (more iterations, larger batches) rather than
   the CPU-constrained, wall-clock-limited run this analysis is based on --
   this is exactly the kind of small-scale-vs-large-scale question the
   starter guide's own honesty flags anticipated.

## Checkpoint selected for Phase B

**Iteration 135 (the final checkpoint)** is selected as the model to carry
into Phase B's circuit-identification work. Rationale: since no single,
unambiguous "post-transition" checkpoint exists in this run (per the
analysis above), the most defensible choice is the most-trained model
available, rather than picking an earlier point that would require claiming
a specific transition happened there when the evidence for that is weak. If
Phase B's patching results are hard to interpret, revisiting checkpoint
selection (e.g. trying iteration 90, where both intervention and
counterfactual accuracy show local peaks and LLC is at its plateau's local
maximum of 17.92) is a reasonable fallback.
