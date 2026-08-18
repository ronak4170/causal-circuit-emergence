# Phase A v2 Conclusion: RQ1 Fix Follow-up

**Motivation:** `results/phase_a_conclusion.md` found RQ1's staged-emergence claim
untestable for the association rung specifically, because the original warm-up (400
examples, 3 epochs) saturated association to ~100% accuracy before RFT even began. This
follow-up reruns Phase A with two changes: (1) a 20x weaker warm-up (`warmup_sft_v2.py`:
60 examples, 1 epoch — verified locally to leave association at ~23-27% accuracy with the
answer format still reliably learned), and (2) a doubled per-iteration batch (8
questions/rung instead of 4, halving the accuracy-quantization step size).

## Setup

Trained on Colab GPU, 150 iterations, checkpoints every 10 iterations (15 checkpoints: 0,
10, ..., 140). Survived one lost run to a Colab runtime reset (recovered by symlinking the
checkpoint directory directly into Google Drive before the second attempt, so training
output persists independently of the ephemeral Colab session).

## Result 1: the behavioral fix worked

Using the identical jump-detection method as the original Phase A analysis (10-iteration
rolling mean, >20pp threshold sustained through the rest of training — same code, same
thresholds, for direct comparability): **association now shows a real, isolated,
detected jump at iteration 81** — dipping to a noisy ~0.30-0.40 range through iteration
~70, then rising to a sustained ~0.55-0.70 from iteration 80 onward. Neither intervention
nor counterfactual show a detected jump by the same criterion, though intervention shows a
visually similar (weaker, sub-threshold) uptick in the same region.

This is a materially different, more theoretically coherent picture than the original run,
where association could show no jump at all (already at ceiling) while intervention and
counterfactual showed jumps instead — backwards from Pearl's predicted ordering
(association first). Here, association is the one rung that clears the detection
threshold, consistent with it being the easiest/first-to-emerge rung.

## Result 2: the LLC signal still doesn't track it — if anything, less than before

The LLC trajectory across all 15 checkpoints is flat within noise: 19.3 (step 0) drifting
to ~18.1-18.4 for most of the back half of training, with per-checkpoint standard
deviations (1.4-1.9) comparable to or larger than the entire range of variation across all
15 points. Visually and statistically, **no transition is detectable anywhere in this LLC
trajectory** — critically including no signal anywhere near iteration 80, where the
association jump actually happens.

This is a weaker LLC-behavior correspondence than even the original Phase A run showed. The
original run at least had one clear, large, twice-replicated LLC transition (10→~140,
completing by iteration 15-30) — real, even though it didn't align with any specific rung's
accuracy jump. This v2 run's LLC trajectory doesn't show any transition at all to not-align
with.

## Interpretation

Combining both findings: fixing the warm-up confound produced a cleaner, more
theoretically-satisfying *behavioral* emergence story (a real, isolated association jump),
but did not produce — and if anything worked against — evidence of a corresponding
*internal* (LLC) signal. Two honest candidate explanations:

1. **LLC genuinely isn't sensitive to this specific kind of capability change.** The
   association jump here may reflect a comparatively subtle shift in the model's
   computation (e.g. gradual weight adjustment within an already-established circuit)
   rather than the kind of qualitative, complexity-changing reorganization LLC is
   theoretically suited to detect (per the grokking-literature framing LLC comes from).
2. **Checkpoint/estimation resolution is still a limitation.** 15 checkpoints spaced every
   10 iterations, each estimated with substantial noise (std ~1.5-1.9 against a total
   observed range of ~1.3), may simply be too coarse and too noisy to resolve a transition
   even if one exists. A denser checkpoint grid specifically around iterations 70-90, and/or
   more SGLD chains/draws per checkpoint to shrink the per-estimate error bars, would be
   needed to rule this out before concluding LLC is genuinely insensitive here.

**Overall RQ1 status after both the original and this follow-up run:** across two
independently-designed experiments, LLC has shown exactly one real, replicated transition
(in the original run, aligned with nothing specific), and zero detectable transitions in
this more carefully-designed follow-up (where a real, clean behavioral transition exists to
potentially align with). The most defensible conclusion is that **this project's LLC-based
approach to RQ1 has not found convincing evidence that phase transitions in the loss
landscape track staged rung-specific emergence** — a genuine, informative negative result,
not a design failure, since the follow-up specifically addressed the confound that could
have explained away the original run's weak result.

## What would be needed to resolve this further

1. Denser LLC checkpointing specifically around iterations 60-100 (where the actual
   behavioral transition occurs in this run) to rule out coarse checkpoint spacing as the
   explanation for the flat LLC trajectory.
2. More SGLD chains/draws per checkpoint to reduce the per-estimate noise (currently
   std ~1.5-1.9, large relative to the observed range).
3. As flagged in the publication roadmap: consider whether LLC is the right tool for this
   specific claim at all, versus directly tracking circuit-level changes (e.g. re-running
   Phase B's patching analysis at multiple Phase A checkpoints) as a more targeted
   alternative measure of "when does the mechanism change."

## Status for the paper

Per the publication roadmap's Option B framing (lead with the verified consistency-vs-
correctness finding; RQ1 as a supporting sub-study, not load-bearing): this result belongs
in that supporting role. It shows real methodological rigor (identifying and fixing a
genuine confound) and an honest negative result, but does not salvage RQ1 as a headline
claim — consistent with the roadmap's original recommendation to not treat RQ1 as
load-bearing regardless of how this follow-up came out.
