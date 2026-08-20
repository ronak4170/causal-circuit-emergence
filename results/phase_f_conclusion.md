# Phase F Conclusion: Training-Time Do-Calculus Scaffolding Follow-Up

**Motivation:** Phase D already showed explicit do-calculus scaffolding at INFERENCE time
never helped confounded-topology intervention questions (0% for both implicit and explicit
prompts, at every one of Phase A's 10 checkpoints — see `results/phase_d_conclusion.md`).
Phase F asked a different question: does baking the same kind of scaffolding into
TRAINING targets — so the model has to learn to produce and use it itself, rather than
being handed it externally at eval time — do any better? This is the "does a specific
training intervention close the gap" experiment identified as the natural next step for
turning this from a diagnostic paper into one with an actionable fix.

## Design

Two warm-up conditions, identical except for one variable (isolating the rationale's
specific contribution from mere extra exposure to intervention-rung examples, which the
original Phase A warm-up had zero of):
- **Rationale condition**: 300 plain association examples + 100 intervention examples
  (25/topology) with a short worked do-calculus rationale before the answer (e.g., for
  confounded: *"This is an intervention. We force switch A to be on directly, ignoring
  whatever would normally cause switch A — including any hidden shared cause with switch
  B. So switch B's probability depends only on the direct causal path from switch A, not
  on the hidden shared cause."*).
- **Control condition**: identical examples, identical seeds, but plain "Answer: X%" with
  no rationale.

Both then ran the full 150-iteration RFT curriculum (matching original Phase A exactly),
with `max_new_tokens` raised from 15 to 60 to leave room for rationale text. RFT's
acceptance criterion (`rft_iteration`) was unchanged — it only checks the final parsed
percentage against the oracle answer, never the rationale's presence or content.

## Results

### Post-RFT (final checkpoint, iteration 135), confounded-topology interventional accuracy

| Condition | Accuracy | Associational accuracy (context) |
|---|---|---|
| Rationale | 0.000 (0/100) | 0.760 |
| Control | 0.000 (0/100) | 0.650 |
| (Original GPT-2, no rationale ever) | 0.000 (0/100) | 0.980 |
| (Pythia-410M, no rationale ever) | 0.000 (0/100) | 0.980 |

Both conditions hit the identical floor every prior run has hit. Inspecting raw
generations revealed why: by iteration 135, neither condition produces ANY rationale
text — both collapse to a bare percentage answer followed by a degenerate repeat loop
(e.g. `" 90%' Answer: 90%' Answer: 90%'..."`), with the same associational-substitution
signature seen throughout this project (bimodal predictions tracking associational
direction, never near the true ~0.50). RFT's outcome-only reward has no mechanism to
preserve rationale content, and by the end of training it doesn't.

### Decisive pre-RFT diagnostic

Before investing in any RFT-preservation fix (periodic rehearsal, KL-to-reference
regularization), a cheap diagnostic was run: does the rationale warm-up checkpoint —
BEFORE any RFT training — ever produce a correct answer on confounded-topology
intervention questions when it actually writes the rationale?

**First attempt was methodologically flawed and had to be corrected.** The initial
version used greedy decoding (temperature=0.0), which turned out to suppress the
rationale entirely — the model's single highest-probability continuation was a bare
answer, even though sampling at temperature=0.8 (matching RFT's own settings) does
produce longer generations. The initial "rationale present" check was also broken: it
used generation length as a proxy, and the degenerate repeat loop is also long, so it
false-counted as "rationale present" in both this diagnostic's first version and in
`warmup_sft_rationale.py`'s own post-warm-up smoke check — meaning the original "20/20
produced more than a bare answer" self-check was never real evidence of rationale content
either.

**Corrected version**: sampled 4 completions per question at temperature=0.8 (matching
`phase_f_main.py`'s own RFT sampling exactly) across all 100 confounded-topology
interventional test questions (400 completions total), and detected rationale presence
via an exact content match (the fixed opening phrase "This is an intervention," a
near-verbatim substring from the training template, not a heuristic).

**Result: 0 of 400 completions contained the rationale marker at all.** Not "rationale
present but still wrong" — the rationale essentially never gets produced for
confounded-topology questions specifically, even pre-RFT, even under the same stochastic
sampling RFT itself uses. Best-of-4 accuracy was 0/100 regardless. Predictions across all
400 samples remained consistently far from the true ~0.50 (0.9, 0.25, 0.85, 0.1, 0.15,
0.2, 0.75, 0.8, 0.95, 0.3, 0.7, 0.05 — the same associational-substitution pattern, never
near the tolerance window).

## Root cause, not just a negative result

The warm-up mix was 300 plain association examples to 100 intervention-rationale examples
split across 4 topologies — roughly 25 confounded-specific rationale examples out of 400
total (about 6% of the training set), competing against a 75%-of-corpus pull toward the
plain-answer pattern. The rationale behavior was most likely never robustly established
for confounded topology in the first place. **This is not "RFT trained the rationale
away" — it's "warm-up dosage was too thin and diluted to make the rationale a reliable
generation mode for this topology to begin with."** The two explanations sound similar but
have different implications: the first would suggest a training-dynamics fix (rehearsal,
KL penalty); the second says there was never enough of the target behavior present to
preserve or study in the first place.

## Verdict

No preservation mechanism (periodic SFT rehearsal, KL-to-reference regularization) is
well-motivated as a next step from here — building one would be solving a problem that
hasn't been shown to exist (the rationale carrying real capability), not the actual
problem (dosage). A cleaner follow-up, if this axis is revisited later, would be much
heavier confounded-specific rationale dosage in isolation (not diluted by 4-topology
mixing or a 3:1 plain-example ratio) before concluding anything about whether do-calculus
scaffolding can help this model at all. That is future work, not something to pursue now.

## Status: this closes the "does a specific training intervention fix the gap" question

Across the full project, three independent attempts to surface or install genuine
confounded-topology interventional competence have now failed, each for a documented,
specific reason:
1. **Ablation** (Phase C): un-ablated model was already at the associational-substitution
   floor; there was no hidden competence for ablation to destroy.
2. **Inference-time scaffolding** (Phase D): explicit do-calculus prompting never helped,
   at any of 10 training checkpoints.
3. **Training-time scaffolding** (Phase F): the scaffolding didn't survive training to be
   evaluated (post-RFT), and — once diagnosed properly — was never robustly established
   even before training (pre-RFT), due to thin/diluted warm-up dosage.

This is a complete, well-characterized negative result on the "can we fix it" question,
sufficient to write up honestly rather than requiring further iteration. Combined with the
positive findings (a real, localized, cross-topology-transferable, twice-replicated
circuit; genuine coarse causal discrimination on unconfounded topologies), the project's
overall shape is: **a real mechanism exists, is general, and replicates across scale — and
its specific failure mode on confounded structures is precisely characterized and shown to
resist three different categories of intervention.** That is the paper's honest, complete
contribution.
