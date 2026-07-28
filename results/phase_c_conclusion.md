# Phase C Conclusion: RQ3 (Ablation Signature)

**Pre-registration:** written and pushed BEFORE any Phase C experiment code existed
(commit `e2ae1c7`, 2026-07-28 17:12:18 -0400), with a test-set-construction addendum
(commit `6cb2884`) added after discovering the naive test set was empty but BEFORE any
ablation was run or scored. Both committed to the public repo; see `PREREGISTRATION.md`
for the exact predictions and thresholds reproduced below.

**Circuit tested:** L7H5, L10H7, L8H11 (attention heads); MLP layers 7, 9, 10, 11 —
identified and validated (random-circuit baseline, 100th percentile) in Phase B.
Iteration-135 checkpoint. **Ablation method:** mean ablation, as pre-registered.
**Test set:** confounded topology only, n=100 interventional + n=100 count-matched
associational questions (restricted from the original 4-topology plan — see the
pre-registration addendum for why chain/fork/collider cannot support this test at all).

## Pre-registered predictions and results

| Prediction | Threshold | Result | Verdict |
|---|---|---|---|
| P1: Directional bias | >50% closer-to-associational AND p<0.05 AND effect size ≥0.65 | 50.0% closer-to-associational (exactly 50/50 split), p=0.540 | **NOT SUPPORTED** |
| P2: Selectivity | Associational accuracy within 10pp of un-ablated | 98.0% → 79.0% (19pp drop) | **NOT SUPPORTED** |
| P3: Effect asymmetry | Interventional drop ≥ 2x associational drop | Interventional: 0.0% → 0.0% (0pp drop); ratio undefined/degenerate | **NOT SUPPORTED** |

All three pre-registered predictions failed as stated. Per the pre-registration's own
interpretation table, this is a legitimate, informative negative result — but the raw
numbers alone don't tell the real story here, which is more specific and more interesting
than "the predictions failed."

## What actually happened (the story behind the numbers)

Inspecting the raw predictions (not just the pass/fail summary) reveals something the
three-prediction framework wasn't designed to capture:

**The un-ablated model was already at 0% interventional accuracy, exhibiting the
theory-predicted associational shortcut *before any ablation*.** On confounded-topology
intervention questions, the un-ablated model outputs exactly one of two values -- 0.85 or
0.10 -- and which one it picks tracks the *associational* truth's direction (high vs. low)
with apparent perfect fidelity, never anything near the true interventional answer (which
is always ~0.50, since `do(A)` genuinely severs the confound and B's distribution collapses
to its unconditional marginal regardless of the forced value of A). This is Pearl's
predicted failure mode -- but as this model's baseline competence on this specific
topology, not as something Prediction 1 could detect, since there was no room left for
ablation to push the model *further* toward an already-100%-associational baseline.

**Ablation did not push the model further toward association -- it collapsed the model's
output to a single constant.** After ablating the circuit, the model outputs exactly 0.10
for **all 100** interventional questions, regardless of whether the true associational
direction was high or low. This is a qualitatively different failure than "reverting to
the associational shortcut": it's a total loss of input-sensitivity on this question type.
The exact 50/50 split in Prediction 1's classification is a geometric artifact of this
collapse -- 0.10 happens to land closer to the ~0.50 interventional truth for roughly half
the test questions (whichever had a low associational truth) and closer to the
associational truth for the other half (whichever had a high associational truth,
where 0.10 is still "closer" to a value like 0.05-0.13 than a constant near the
interventional 0.50 would suggest -- but note these are the SAME low-associational-truth
questions where the un-ablated model was already answering ~0.10, so for that half nothing
really changed). This is not balanced, unbiased directional error in any meaningful sense;
it is output collapse that happens to score as 50/50 by coincidence of where the true
answers land relative to one fixed number.

**On associational questions, ablation caused partial miscalibration, not collapse.** The
associational-question ablated outputs remain bimodal (0.10 / 1.0, vs. 0.10 / 0.85
un-ablated) -- the model still discriminates high vs. low associational truth after
ablation, it just shifts its "high" answer from 0.85 to 1.0, which falls outside the 0.10
tolerance for some (not all) true values in that cluster, producing the 98%→79% accuracy
drop. This is a real, measurable selectivity failure (P2), but a much milder one than the
complete collapse seen on interventional questions.

## Revised interpretation

The pre-registered framework assumed un-ablated interventional accuracy would be
meaningfully above zero, giving ablation room to shift outputs in a specific direction.
That assumption was wrong for this topology: the model's baseline competence on confounded
intervention questions was already at floor, already matching the associational shortcut
essentially completely. This is itself a finding -- and arguably a more interesting one
than a clean P1 pass would have been, since it says the model's Phase A / Phase B
"interventional competence" on the *hardest, most genuinely confounded* topology may not
be real competence at all, but a systematic associational substitution that was present
throughout training, undetected by Phase A's coarser accuracy tracking (which pooled all
four topologies together) and not specifically tested by Phase B's transfer experiment
(which used fresh pairs but did not separately check whether the *unablated* model's
confounded-topology answers were correct in the interventional sense, only whether
patching restored the clean-vs-corrupt distinction — see the caveat added below).

What ablation *does* show clearly: it destroys the circuit's contribution to input-specific
computation on this question type (100% output collapse), while associational processing
degrades more mildly (bimodality preserved, just recalibrated). This is consistent with
the circuit playing SOME role in producing input-sensitive answers to confounded
intervention questions specifically -- just not the role Pearl's directional hypothesis
predicted. The honest characterization, using the pre-registration's own interpretation
table: **P1 fails in a way that suggests the circuit's function is not "compute the
intervention vs. association distinction correctly" but something more like "maintain
input-sensitivity for this question type at all" -- and the baseline (pre-ablation)
associational bias this analysis surfaced is a separate, arguably more important, finding
about the underlying model's actual competence than the ablation signature test itself.**

## A new caveat this analysis surfaces for Phase B

Phase B's cross-topology transfer numbers (90-98% restoration) measured whether patching
the clean run's activations into the corrupted run restored the *clean-vs-corrupt
distinction* -- they did not check whether the clean run's own answer was interventionally
*correct*. Given what Phase C found (0% baseline interventional accuracy on confounded
topology), it's possible Phase B's "clean" runs were themselves confidently and
consistently wrong (matching the associational shortcut) rather than confidently right --
patching could restore a *consistent* wrong answer just as easily as a correct one. Phase
B's transfer result is not invalidated by this (it was always a claim about mechanistic
consistency/localization, not correctness), but it should not be read as evidence the
circuit computes CORRECT interventional answers -- only that it computes SOME consistent,
localized, input-dependent computation that patching can move around. This distinction
should be flagged prominently in any future write-up.

## What would resolve this

1. Directly measure un-ablated interventional accuracy, per topology, before running any
   future ablation test -- this should have been a pre-condition check in Step 1
   (analogous to Phase B's checkpoint-verification step) and would have caught the floor
   effect before spending the pre-registration on a test that couldn't discriminate.
2. Re-run Phase C's directional test on a topology/checkpoint combination where baseline
   interventional accuracy is meaningfully above zero, if one exists -- the mixed-topology
   55% accuracy Phase B measured (n=20, all 4 topologies pooled) suggests chain/fork/
   collider questions ARE sometimes answered correctly, even though those topologies can't
   test directional bias (no assoc/interv divergence). A different, real experimental
   design question for future work: is there ANY topology with both (a) real
   assoc/interv divergence and (b) non-floor baseline interventional accuracy? Confounded
   has (a) but apparently not (b).
3. Investigate why the un-ablated model fails so completely on confounded intervention
   questions specifically, given Phase A's logged intervention accuracy (pooled across
   topologies) was 55-100% at various points in training -- was confounded systematically
   under-represented in what the model actually got right during RFT?

## Checkpoint / circuit status for later phases

The circuit (L7H5, L10H7, L8H11 heads; MLP 7/9/10/11) remains the object of interest for
any future work, but Phase C's result means it should not be described as "the
intervention-vs-association circuit" -- that specific characterization was tested and not
supported. "A circuit whose ablation collapses input-sensitivity on confounded-topology
questions" is the more defensible description going forward.
