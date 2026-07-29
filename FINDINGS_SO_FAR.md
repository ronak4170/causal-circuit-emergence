# Findings and Conclusions So Far (Phase 0, Phase A, Phase B, Phase C, Phase D)

**Last updated:** after Phase D completion, before Phase E.

This document synthesizes what has actually been found across the five completed
phases, separated from the phase-by-phase working notes in `docs/phase0_setup.md`,
`results/phase_a_conclusion.md`, `results/phase_b_conclusion.md`,
`results/phase_c_conclusion.md`, and `results/phase_d_conclusion.md`. Read those for
full methodological detail; this file is the consolidated "what do we actually know
now" summary.

---

## 1. Phase 0 — Tooling Validation (not a research result, a sanity check)

**Purpose:** confirm the LLC-estimation and activation-patching tooling work
correctly on two already-published results, before trusting them on the novel
research question.

**Findings:**

- **Grokking reproduced cleanly.** A 1-layer transformer trained on modular
  addition showed the textbook delayed-generalization curve: train accuracy
  saturated to 100% by step 1000, test accuracy lagged (0.7% → 18%) then jumped
  sharply to ~99% by step 2200 and 100% by step 2600.
- **LLC showed a real, but not textbook, signal.** The single largest LLC
  transition (10.0 → ~140) completed by step 1000 — i.e. it tracked
  *train-accuracy saturation*, not the later *generalization* jump (~step
  2000–2600), which is the more commonly reported alignment in the literature.
  What LLC *did* track cleanly: three separate transient training
  destabilization events (steps ~2400, ~6000, ~11400), where both train and
  test accuracy briefly collapsed and LLC dipped in near-perfect lockstep each
  time. This is a genuine, reproducible correlation between LLC and training
  dynamics — just a different one than the "LLC spikes at the grokking moment"
  story.
- **IOI circuit patching found genuine overlap with published results.** Top
  patched heads on GPT-2 small included **L9H9** and **L9H6** — two of the
  three canonical Name Mover Heads reported in Wang et al. 2022 (L9H9, L9H6,
  L10H0). The patching heatmap showed a small, structured set of heads
  (concentrated in layers 8–11), not a diffuse pattern.

**Verdict:** 2 of 3 checkpoints matched published results cleanly; the LLC
timing mismatch was reported honestly rather than glossed over. Tooling judged
trustworthy enough to proceed.

**Environment note carried forward into all later phases:** this project runs
on a single Apple Silicon Mac with no CUDA GPU. MPS was deliberately excluded
from all device auto-detection because TransformerLens documents it as
possibly producing "silently incorrect results" — a correctness risk not worth
the speed gain. CPU was used locally throughout, with Google Colab used
opportunistically for GPU-bound steps (see Phase A).

---

## 2. Phase A — RQ1: Staged Emergence (partial support, weak — leaning falsification)

**Setup:** GPT-2 small, warm-started with association-only SFT (required to
solve reward sparsity), then RFT across a curriculum spanning all three Pearl
rungs (association/intervention/counterfactual) and all four DAG topologies
(chain/fork/collider/confounded), 150 iterations. Run twice independently: once
locally (CPU) and once on Colab (GPU, different seed) as a replication check.

**Findings:**

- **Association's emergence was untestable by design.** The warm-up step
  needed to avoid reward sparsity (Step 2.3's explicit requirement) saturated
  association accuracy to ~100% *before RFT even began*. There was no room
  left for it to "emerge" — this is a design confound in this specific run,
  not evidence against Pearl's predicted ordering.
- **Intervention and counterfactual showed detected "jumps," but neither was
  clean.** Using a pre-specified (though not strictly blinded — see honesty
  note below) 20-percentage-point / 10-iteration rolling-mean criterion:
  intervention jumped at iteration 10 then settled into a noisy 0.4–0.7
  plateau; counterfactual jumped at iteration 86 after a trough as low as 0.15,
  but dipped again afterward before a final rise. Neither looks like a sharp,
  well-isolated phase transition — both look more like noisy drift, which is
  partly a real limitation of only sampling 4 questions/rung/iteration
  (accuracy is quantized to {0, .25, .5, .75, 1.0}).
- **LLC showed one clear, replicated transition, but only one.** Both the
  local CPU run and the independent Colab GPU run showed the same shape: LLC
  rises sharply in the first 15–30 iterations (local: 15.4→17.7; Colab:
  16.5→19.4), then plateaus with only minor fluctuation for the rest of
  training. This replication across two independently-seeded, independently-
  hardware'd runs is real evidence the *shape* is not noise. But there is no
  second LLC transition anywhere later in training — critically, nothing
  aligns with counterfactual's iteration-86 jump.
- **A real methodological bug was caught and fixed before it corrupted
  results.** An early version of the jump-detection smoothing used
  zero-padded convolution, which manufactured a fake simultaneous "jump at
  iteration 10" across all three rungs — an artifact of the boundary
  condition, not a finding. Caught by inspecting the plot before writing
  conclusions, and fixed with a proper boundary-aware rolling mean.

**Verdict:** the honest reading is *partial support, weak, leaning toward
falsification* — one real LLC transition exists and is replicated, but the
three-rung, cleanly-ordered staged-emergence story that would count as "full
support" is not what the data show. Written up in full, including the
disclosed non-blinding of the jump-detection criterion (the raw per-iteration
numbers were watched live during the ~2.5-hour training run before the
detection script was formally written), in `results/phase_a_conclusion.md`.

**Checkpoint selected for later phases:** iteration 135 (the final
checkpoint) — chosen because no single unambiguous "post-transition" point
existed to justify picking an earlier one.

---

## 3. Phase B — RQ2: Structural Generalization of the Circuit (full transfer supported, with a caveat)

**Setup:** activation patching (single-forward-pass, single-token logit-diff
metric — GPT-2's BPE tokenizes every 5%-rounded percentage as one token, which
made a cheap, IOI-style metric possible) across all 144 attention heads and 12
MLP layers, averaged over 60 clean/corrupt intervention-question pairs spanning
all 4 topologies, on the iteration-135 checkpoint (first verified to score 55%
on a fresh 20-question intervention eval — well above chance).

**Findings:**

- **A small, sparse candidate circuit emerged cleanly.** One head, **L7H5**,
  dominates (normalized patching effect 0.505 — more than 3x the next
  largest). Two secondary heads (L10H7: 0.150, L8H11: 0.123) and four
  later-layer MLPs (layers 7, 9, 10, 11: effects 0.106–0.374) round out the
  circuit. 141 of 144 heads and 8 of 12 MLP layers showed near-zero effect —
  this is a genuinely localized result, not the diffuse "everything matters a
  little" pattern that would have been a weaker finding.
- **The circuit transfers almost completely across all four topologies.**
  Patching *only* these 7 components (not the full grid) on fresh held-out
  question pairs restored 90–98% of clean-run behavior in every topology
  tested: chain 0.981, fork 0.974, collider 0.904, confounded 0.974. Notably,
  collider and confounded — the two topologies structurally different enough
  to make transfer a real test (see the Phase A design notes on why chain/fork
  alone would be too easy) — transferred just as well as the more similar
  chain/fork pair.
- **A design bug was caught before it wasted the whole patching run.** The
  first attempt at building confounded-topology clean/corrupt pairs used an
  (A, B) intervention pair, which correctly gave near-zero divergence between
  clean and corrupt — because forcing A via `do()` genuinely severs the U
  confound, so B is unaffected either way. This is *scientifically correct*
  behavior (a nice independent confirmation the causal simulation is right),
  but it made that pair useless for the minimal-pair patching design, which
  needs a real divergence to detect. Fixed by switching to the (A, C) pair,
  which has a real direct causal edge.
- **The circuit is not an artifact of the task being narrow and templated.**
  A natural worry: since the answer is a single teacher-forced percentage
  token, maybe *any* small set of late-layer components would restore most
  of the clean-run answer regardless of which ones. Tested directly: 25
  random circuits of the same size/composition (3 random heads + 4 random
  MLP layers) restored a mean of only 26.7% (std 17.4%, max 53.4%) on the
  same held-out pairs, vs. the candidate circuit's 95.8% — the candidate beat
  all 25 random circuits (100th percentile). This rules out the
  task-narrowness explanation and substantially strengthens confidence that
  L7H5 and its supporting components are doing something specific, not just
  "whatever late-layer components happen to be patched."

**Verdict:** closest to "full transfer" support for RQ2 — the model appears to
have learned one shared, topology-general intervention-computation mechanism
rather than separate topology-specific circuits. The important caveat,
disclosed rather than hidden: the circuit was identified from a batch that
already included each topology's own data, so this shows generalization to
*new questions within topologies partly seen during identification*, not
transfer to a topology withheld entirely from circuit-finding. A stricter
leave-one-topology-out version is listed as a concrete follow-up in
`results/phase_b_conclusion.md`, not yet run.

**Circuit + checkpoint carried into Phase C:** L7H5, L10H7, L8H11 (heads);
MLP layers 7, 9, 10, 11; iteration-135 checkpoint.

---

## 4. Phase C — RQ3: Ablation Signature (all 3 predictions not supported — but with the
most interesting finding of the project underneath)

**Setup:** three predictions pre-registered and pushed to GitHub (commit `e2ae1c7`,
timestamped before any Phase C code existed) — directional bias, selectivity, and effect
asymmetry, each with exact thresholds. Mean ablation of the Phase B circuit, tested on 100
confounded-topology intervention questions + 100 count-matched associational questions.

**A test-set construction problem surfaced first, before any ablation ran.** The naive
approach (reusing Phase B's variable-pair choices) produced zero usable questions in all 4
topologies. Root cause: association and intervention answers only genuinely diverge, for
the variable pairs the model was actually trained on, in the confounded topology (0.41
divergence) — chain/fork/collider were deliberately built in Phase A as *unconfounded
control cases* with ~0 divergence. This forced narrowing the test set to confounded only.
Documented as a pre-registration addendum (commit `6cb2884`) before any ablation was run
or scored, preserving the pre-registration discipline.

**All three predictions failed as stated:**

| Prediction | Result |
|---|---|
| P1 (directional bias) | Exactly 50/50 split, p=0.540 |
| P2 (selectivity) | Associational accuracy 98%→79% (19pp drop, exceeds 10pp tolerance) |
| P3 (asymmetry) | Interventional accuracy 0%→0% (floor effect, no drop to measure) |

**But inspecting the raw predictions (not just the pass/fail summary) revealed the real
story:** the *un-ablated* model was already at 0% interventional accuracy on confounded
questions, answering in the associational direction 100% of the time *before any ablation
at all* — Pearl's predicted associational-shortcut failure mode was already this model's
baseline behavior on its hardest topology, leaving no room for Prediction 1 to detect a
*shift*, since there was nowhere further "associational" to shift to. Ablation didn't push
the model more toward association — it collapsed the model's output to a single constant
(0.10) for all 100 interventional questions regardless of the true answer's direction,
while associational-question outputs stayed bimodal but shifted (0.85→1.0), producing the
milder selectivity failure. The exact 50/50 split in Prediction 1 is a geometric artifact
of this collapse, not evidence of balanced, theory-neutral errors.

**Verdict:** a genuine negative result on all three pre-registered predictions, but one
that surfaces a more important finding than any of them would have individually: this
model's apparent interventional competence on its hardest, most genuinely confounded
topology may not be real competence at all — Phase A's pooled-topology accuracy logging
and Phase B's patching-based transfer test were both structured in ways that couldn't have
caught this, since neither directly checked whether the *un-ablated* model's confounded
answers were correct in the interventional sense.

**New caveat for Phase B, surfaced by this analysis:** Phase B's 90–98% cross-topology
transfer numbers measured mechanistic *consistency* (does patching restore the clean run's
answer), not *correctness* (was the clean run's answer actually right). Given Phase C's
finding, it's possible Phase B's "clean" runs were themselves confidently answering with
the associational shortcut rather than the correct interventional answer — patching can
restore a consistent wrong answer as easily as a correct one. This doesn't invalidate
Phase B's localization/transfer claim, but it means that claim should not be read as
evidence the circuit computes correct answers.

**Full write-up:** `results/phase_c_conclusion.md`. **Circuit's revised characterization**
going forward: not "the intervention-vs-association circuit" (tested, not supported) but
"a circuit whose ablation collapses input-sensitivity on confounded-topology questions."

---

## 5. Phase D — RQ4: Behavioral Correlate (effectively falsified — no gap existed to
close, independently corroborating Phase C)

**Setup:** matched implicit/explicit intervention prompt pairs (1 implicit + 3 explicit
do-calculus-scaffolding variants), built from each topology's actual trained variable pair
(applying the same in-distribution lesson Phase C learned the hard way). Evaluated at all
10 available Phase A checkpoints, n=15 questions/topology, with Wilson-score confidence
intervals to distinguish real gaps from small-sample noise.

**Almost none of the apparent gaps were statistically real.** Of 40 (checkpoint x
topology) cells, only 1 -- chain at iteration 90 -- had non-overlapping confidence
intervals. Every other apparent gap, including some as large as 33-53 percentage points in
raw terms, had overlapping CIs given the small sample size and cannot be distinguished from
noise. This is itself a useful methodological point: eyeballing raw point-estimate
"gaps" without confidence intervals would have produced a much noisier, more
overinterpreted-looking story.

**The one statistically real effect was the opposite of the hypothesis: scaffolding hurt.**
At iteration 90, explicit prompts drove chain accuracy from 53% down to 0% -- exactly the
"scaffolding may confuse the model" failure mode the phase's own design guide flagged as
possible, observed directly rather than merely anticipated.

**Confounded topology -- the only topology diagnostic of genuine intervention-vs-
association reasoning -- scored exactly 0% for BOTH implicit and explicit prompts, at
every single checkpoint, with zero variance.** Not even the most heavily scaffolded
variant (explicit do-operator notation, written out directly: `P(target | do(var=val))`)
ever rescued a single question. This is a floor, not parity from competence.

**This independently corroborates Phase C's central finding via a completely different
method.** Phase C ablated the model's internals and found 0% un-ablated interventional
accuracy on confounded questions. Phase D never touches the model's internals at all --
only the prompt -- and finds the identical 0% floor, regardless of how much explicit
causal-reasoning scaffolding is provided, at every point across training. Two independent
probes (one mechanistic, one purely behavioral) converge on the same limitation. This is
exactly the "two independent thermometers" logic the phase was designed around -- just
converging on a genuine incapacity rather than the hoped-for capacity-that-internalizes
story.

**Two follow-up checks (run after the user asked how to strengthen this result) resolved
the main open questions:**

- **Length confound, resolved:** a length-matched, content-free `neutral_control` condition
  (246 vs. explicit's 245 mean characters) was tested across all 10 checkpoints.
  It matched `implicit` **exactly** at every single checkpoint, including the two where
  `explicit` had diverged — cleanly attributing both real effects to scaffolding content,
  not prompt length.
- **"Is chain/fork/collider's accuracy genuine?" — yes.** A discrimination test (n=40/
  topology at the final checkpoint) checked whether the model's predictions actually track
  the true causal effect and correctly flip direction with `do_value`, rather than just
  landing within tolerance of a memorized constant. Correlation between predicted and true
  answers was 0.985–0.998, with correct directional discrimination in all three
  topologies. Caveat: predictions are coarse/binary-ish (two clustered output values per
  topology), not fully graded probability estimates — real competence, but not
  fine-grained continuous reasoning.

**This sharpens the overall picture substantially.** It is not "the model has no causal
competence" — Follow-up 2 rules that out for unconfounded topologies. It is specifically:
**genuine, verified, direction-correct competence on chain/fork/collider, and a complete,
floor-level absence of that same competence on confounded** — the one topology that
actually requires distinguishing intervention from association. This is a materially more
informative conclusion than the original Phase D write-up alone supported.

**Full write-up:** `results/phase_d_conclusion.md`. **Cross-phase integration figure:**
`results/cross_phase_integration.png` (Phase A accuracy + LLC + Phase D gap, one shared
training-iteration axis). **Discrimination test scatter:**
`results/phase_d_discrimination_scatter.png`.

---

## 6. Cross-Cutting Deductions (things learned across all four phases)

1. **Small-model, small-batch RL settings are noisy enough that "phase
   transition" claims need real skepticism.** Phase A's per-iteration
   accuracy is quantized and volatile; only signals that replicate across
   independent runs (like the LLC early-transition shape) should be trusted.
   Single-run "jumps" detected by a threshold rule are easy to manufacture by
   accident (see the smoothing-artifact bug) and should be visually inspected,
   not just algorithmically flagged.

2. **LLC's relationship to behavioral change is looser than the canonical
   grokking story suggests, in both settings tested here.** In Phase 0, LLC
   tracked training destabilization events more cleanly than the
   generalization moment itself. In Phase A, LLC showed one real transition
   that didn't correspond to either rung-specific accuracy jump. Neither
   result invalidates LLC as a tool, but both argue against assuming LLC
   transitions map onto whatever behavioral change is currently being
   hypothesized — the correspondence has to be checked each time, not assumed.

3. **Warm-up/curriculum design choices can silently make a research question
   untestable.** The association-only warm-up in Phase A was a reasonable,
   even necessary, engineering fix for reward sparsity — but it had the
   side effect of making RQ1's "association emerges first" claim structurally
   unfalsifiable in that run. Any pipeline that includes a warm-up or
   curriculum stage should be checked for this kind of ceiling effect before
   trusting downstream "no transition detected" conclusions.

4. **Mechanistic localization can be real even when the behavioral training
   signal (Phase A) is messy — but "localized" and "correct" are different
   claims, and it's easy to conflate them.** Phase B found an unambiguous,
   sparse, strongly transferring circuit on the Phase A checkpoint despite
   Phase A's noisy emergence story. That finding stands on its own. But
   Phase C revealed that on the one topology where it could actually check,
   the *un-ablated* model's "correct-looking" behavior was consistently
   wrong in a specific way (matching the associational shortcut, not true
   intervention). Phase B's patching only ever tested whether activations
   could be moved around consistently — never whether the answer being
   moved around was right. The lesson: a strong localization/transfer result
   is evidence of *a* mechanism, not evidence that the mechanism is doing
   the *correct* computation — those need to be checked separately.

5. **Infrastructure bugs are the main practical risk in this kind of pipeline,
   and several were real, not hypothetical.** Across the three phases: a
   token-vs-string slicing bug that would have silently fed the wrong
   "generated answer" into the entire RFT reward signal; a disk-filling
   checkpoint-accumulation bug that crashed a 2.5-hour training run at 40%
   complete; a zero-padding smoothing artifact that manufactured a fake
   simultaneous jump across all three Phase A rungs; and a scientifically-
   correct-but-operationally-useless prompt-pair design in Phase B. All four
   were caught by inspecting actual intermediate outputs (generated text,
   disk usage, plots, raw pair statistics) rather than trusting a script's
   exit code — the general lesson being that in this kind of exploratory
   pipeline, "it ran without crashing" is a much weaker signal than "I looked
   at what it actually produced."

6. **Cross-environment replication (local CPU vs. Colab GPU) has been useful
   twice, not just for speed.** In Phase A, the same LLC shape appearing in
   two independently-seeded, independently-hardware'd runs was the strongest
   evidence in that phase that a real signal exists rather than noise from one
   run. Worth continuing as a habit in later phases when time allows, not just
   treating Colab as a speed hack.

7. **Pre-registration earned its keep in Phase C, and not in the way it's
   usually pitched.** The textbook value of pre-registration is preventing
   post-hoc reshaping of predictions to match messy results. That happened
   here too (all three predictions failed, honestly reported as failures,
   not quietly redefined). But the more concrete value was structural: the
   pre-registration's exact test-set specification forced the empty-test-set
   problem to surface *before* any ablation ran, at a point where fixing it
   was clearly a data-construction issue rather than a suspicious
   after-the-fact adjustment. Without a written, dated specification to
   check the actual test set against, it would have been much easier to
   quietly adjust the topology selection *after* seeing a weak ablation
   result and lose the ability to tell whether that adjustment was principled
   or convenient.

8. **Aggregate accuracy numbers can hide the entire story — inspecting raw
   per-example outputs mattered as much here as in the earlier bug catches.**
   Phase C's pass/fail summary alone ("all three predictions failed") reads
   as a fairly generic negative result. Looking at the actual predicted
   values revealed a specific, interpretable pattern (baseline associational
   collapse, then ablation-induced output collapse to a constant) that
   changes what the negative result *means*. This is the same lesson as
   deduction 5 above (bugs caught by looking at real outputs), extended from
   "catching mistakes" to "extracting the actual finding" — aggregate
   metrics are a starting point for investigation, not a stopping point.

9. **Confidence intervals, not point estimates, should gate every claim about
   a "gap" or "difference" on small samples — Phase D would have told a much
   noisier, more overinterpreted story without them.** With n=15/topology/
   checkpoint, raw point-estimate gaps as large as 33-53 percentage points
   turned out to be statistically indistinguishable from noise once Wilson
   intervals were computed — only 1 of 40 (checkpoint, topology) cells
   survived. A version of this analysis that plotted and interpreted raw
   gaps alone (easy to do, and what the starter code's own plotting
   suggestion would have produced by default) would have read as "the gap
   fluctuates wildly and unpredictably across training," an much weaker and
   more confusing finding than "there is essentially no real gap anywhere
   except one isolated event, and the diagnostic topology is at a floor."
   Small-n behavioral evaluations are exactly where this check matters most,
   and it would have been easy to skip.

10. **Two independent probes converging on the same limitation is strong
    evidence — even when the limitation is a negative finding neither probe
    was originally designed to highlight.** Phase C's ablation (mechanistic)
    and Phase D's prompting (behavioral) shared no methodology, no code path,
    and no data in common, yet arrived at the identical conclusion: 0%
    genuine competence on confounded-topology intervention questions,
    unmovable by either circuit removal or prompt scaffolding. Neither phase
    set out to prove this specifically (Phase C was testing a directional
    ablation signature; Phase D was testing a training-time gap-closing
    story) — the convergence emerged from honestly reporting what both
    experiments actually showed, rather than from designing a single
    experiment to demonstrate it. That's a stronger form of evidence than
    either phase could have produced alone, and it came from following the
    pre-registration/inspect-raw-outputs discipline consistently rather than
    from a specifically-targeted confirmatory test.

---

## 7. What's Still Open Going Into Phase E

- Phase A's counterfactual-rung jump (iteration 86) has no known mechanistic
  correlate yet — Phase B only investigated the *intervention* circuit, per
  RQ2's scope. Whether counterfactual reasoning shares components with the
  intervention circuit found here is untested.
- Phase B's leave-one-topology-out follow-up (a stricter transfer test) has
  not been run.
- Phase B did not measure per-topology baseline (unpatched) accuracy, so
  collider's slightly lower transfer score (0.904 vs. ~0.97–0.98 elsewhere)
  cannot yet be cleanly attributed to circuit-specificity vs. collider
  questions simply being intrinsically harder for the model in general.
- ~~Is the model's 0% interventional competence specific to confounded, or a
  broader shortcut on chain/fork/collider too?~~ **Resolved:** a
  discrimination test confirmed genuine, direction-correct (if coarse)
  competence on chain/fork/collider specifically (r=0.985–0.998); the floor
  is specific to confounded.
- ~~Phase D's length confound was disclosed but not resolved.~~ **Resolved:**
  a length-matched neutral control matched `implicit` exactly at all 10
  checkpoints; both real effects found are attributable to scaffolding
  content, not length.
- Phase A's counterfactual-rung jump (iteration 86) has no known mechanistic
  correlate yet — Phase B only investigated the *intervention* circuit, per
  RQ2's scope. Whether counterfactual reasoning shares components with the
  intervention circuit found here is untested.
- Phase B's leave-one-topology-out follow-up (a stricter transfer test) has
  not been run.
- Phase B did not measure per-topology baseline (unpatched) accuracy, so
  collider's slightly lower transfer score (0.904 vs. ~0.97–0.98 elsewhere)
  cannot yet be cleanly attributed to circuit-specificity vs. collider
  questions simply being intrinsically harder for the model in general.
- Phase C's associational-question ablation effect (0.85→1.0 recalibration)
  is not yet understood mechanistically — still an open, real, measured
  effect worth a closer look if time allows.
- The one statistically real Phase D effect (chain, iteration 90: scaffolding
  driving accuracy from 53% to 0%) has not been replicated across a
  different seed — still worth doing before treating it as more than a
  single-checkpoint curiosity, even though length is now ruled out as the
  cause.
- The discrimination test's "coarse, binary-ish output" observation (two
  clustered values per topology rather than graded probability estimates)
  is itself worth investigating — is this a general property of this
  model's percentage-answering behavior, or specific to unconfounded
  intervention questions?
- Phase E's recursive-training design should track chain/fork/collider's
  *genuine* competence and confounded's *already-absent* competence as two
  distinct baselines, not one undifferentiated "interventional competence"
  — the sharpened Phase D picture makes this distinction available for the
  first time.
