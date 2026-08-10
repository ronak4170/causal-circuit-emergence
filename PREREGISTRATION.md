# Pre-Registration of Hypotheses (Phases A–E)

This document commits each phase's hypothesis and falsification condition **before** the
corresponding experiment is run. The rule for this file: once a phase begins, its hypothesis
above the "Result" line must not be edited. Only the "Result" subsection may be filled in,
and only after that phase's experiment has been run.

---

## Phase A — RQ1: Emergence

**Hypothesis:** The three rungs of Pearl's causal hierarchy (association, intervention,
counterfactual) emerge as separable phase transitions during RL post-training, detectable
via the Local Learning Coefficient (LLC), and emerge in that order (association first,
then intervention, then counterfactual).

**Falsification condition:** No distinguishable LLC transitions align with per-rung
accuracy jumps, or the three rungs emerge simultaneously or out of the predicted order.

### Result (fill in AFTER running the experiment — do not edit the hypothesis above once the corresponding phase begins)

**Partial support, weak — leaning toward falsification.** Full write-up in
`results/phase_a_conclusion.md`; summary:

- Association showed no detectable jump because a required pre-RFT warm-up
  step (added to solve reward sparsity) had already saturated it to ~100%
  before RFT began -- this makes the "association emerges first" claim
  untestable in this run, a design confound rather than evidence either way.
- Intervention and counterfactual each showed a detected jump (iteration 10
  and 86 respectively, by a pre-specified but not strictly blinded
  20pp/10-iteration rolling-mean criterion), but both are noisy plateaus/drifts
  rather than clean discrete transitions.
- LLC showed one clear, replicated transition (across two independent runs,
  local CPU and Colab GPU) in the first 15-30 iterations, then a flat plateau
  for the rest of training -- it does not show a second transition aligning
  with the later counterfactual jump.
- Falsified: the clean, fully-ordered three-stage picture. Not falsified:
  some jump-like structure exists and LLC is not simply flat throughout.
- Checkpoint selected for Phase B: iteration 135 (final checkpoint) -- see
  conclusion doc for rationale and the iteration-90 fallback option.

---

## Phase B — RQ2: Structural Generalization

**Hypothesis:** The circuit responsible for interventional reasoning on one causal-DAG
topology (e.g. a fork) causally transfers via activation patching to a different topology
(e.g. a chain).

**Falsification condition:** Patching the identified circuit's activations across
topologies produces no consistent restoration of interventional accuracy — i.e. the circuit
is topology-specific and does not transfer.

### Result (fill in AFTER running the experiment — do not edit the hypothesis above once the corresponding phase begins)

**Supported (closest to "full transfer"), with a caveat.** Full write-up in
`results/phase_b_conclusion.md`; summary:

- A small candidate circuit (attention heads L7H5, L10H7, L8H11 + MLP layers
  7, 9, 10, 11 -- 7 of 156 total components) was identified via activation
  patching averaged over 60 clean/corrupt intervention pairs across all 4
  topologies at once.
- Patching only these 7 components restored 90-98% of the clean-run answer
  on FRESH held-out pairs in every topology (chain 0.981, fork 0.974,
  collider 0.904, confounded 0.974) -- including collider and confounded,
  the two topologies expected to be hardest to transfer to.
- Caveat: the circuit was identified from a batch that already included each
  topology's own data, so this shows generalization to new questions within
  seen topologies, not transfer to a wholly unseen topology. A stricter
  leave-one-topology-out version is listed as a follow-up.
- Method deviation from the starter spec: one circuit from a mixed-topology
  batch, not separate per-source circuits tested via a source->target
  matrix -- disclosed in the conclusion doc.
- Random-circuit baseline (added after initial write-up): 25 random circuits
  of the same size/composition restored only 26.7% on average (std 17.4%,
  max 53.4%) vs. the candidate circuit's 95.8% -- candidate beat all 25
  (100th percentile), ruling out "any small late-layer circuit would do this
  on such a narrow task" as an explanation.
- Circuit carried into Phase C: L7H5, L10H7, L8H11 heads; MLP layers 7, 9,
  10, 11; same iteration-135 checkpoint.

---

## Phase C — RQ3: Ablation Signature

**Hypothesis:** Ablating the identified circuit causes the model's errors to shift
specifically toward associational-style answers (`P(Y|X)` instead of `P(Y|do(X))`) — a
theory-predicted, directional failure — rather than generic random degradation.

**Falsification condition:** Post-ablation errors are not disproportionately
associational-style (e.g. the error distribution is statistically indistinguishable from
random degradation across answer categories).

### Pre-registered predictions (written BEFORE any Phase C experiment code exists)

**Circuit under test:** L7H5, L10H7, L8H11 (attention heads); MLP layers 7, 9, 10, 11.
Same set identified and validated (random-circuit baseline, 100th percentile) in Phase B.
Same iteration-135 checkpoint.

**Ablation method:** mean ablation (not zero ablation) — each candidate component's
output is set to its mean activation, computed over a diverse set of causal-DAG prompts
spanning all 3 rungs and all 4 topologies (regenerated via `causal_dag_task.generate_instance`
with a fixed seed, analogous to Phase A's training curriculum distribution — the literal
Phase A prompt log was not saved verbatim, so an equivalent freshly-sampled distribution is
used instead; this is disclosed as a minor deviation from "compute means from Phase A's
actual training prompts"). This choice is locked in now and will not be switched to zero
ablation even if mean ablation produces messier results.

**Test set (interventional):** intervention questions from all 4 topologies, target
n=25/topology, seed=42, kept only if `|associational_true - interventional_true| >= 0.15`
(the divergence filter — without it the directional test is undefined). Exact code:
`src/phase_c_reference_answers.py`.

**Test set (associational, for Predictions 2/3):** pure association-rung questions from the
same 4 topologies, count matched to however many interventional questions were actually
collected per topology.

**Tie tolerance for directional classification:** `|d_interventional - d_associational| < 0.02`
counts as a tie, not a win for either side.

**Accuracy tolerance (matching Phase A/B convention throughout this project):** 0.10.

---

**Prediction 1 — Directional bias (main claim).** On the interventional test set, after
ablating the circuit, the ablated model's predicted probability will be closer to the
associational true answer than to the interventional true answer on **more than 50%** of
decidable (non-tie) cases. Tested with a one-sided binomial test (H0: p=0.5, alternative
"greater"). **Supported** requires BOTH statistical significance (p < 0.05) AND a meaningful
effect size (fraction closer-to-associational >= 0.65) — a barely-significant 52% would not
count as genuine support.

**Prediction 2 — Selectivity.** On the pure associational test set, ablated accuracy
(within 0.10 tolerance of the true associational answer) will be within **10 percentage
points** of un-ablated accuracy on the same questions.

**Prediction 3 — Effect asymmetry.** The accuracy drop (un-ablated minus ablated) on the
interventional test set will be **at least 2x** the accuracy drop on the associational test
set.

**Interpretation table (fixed now, not chosen after seeing results):**
- All three supported -> circuit implements a directional, intervention-specific mechanism.
- P1 fails -> circuit matters but doesn't cleanly implement intervention-vs-association;
  errors are more diffuse than a strict do-calculus account predicts.
- P2 fails -> ablation is too broad / circuit serves multiple functions, not
  intervention-specific.
- P3 fails -> circuit is generic to causal-DAG reasoning, not intervention-specific.

This pre-registration checklist is complete: predictions written with exact thresholds;
ablation method specified (mean ablation); test sets specified; about to be committed and
pushed to public GitHub BEFORE any Phase C experiment code is written.

### Addendum (written after `src/phase_c_reference_answers.py` was run once, but BEFORE
any ablation experiment or scoring — this is a test-set-construction fix, not a
result-driven change; no ablated-model output has been observed at this point)

Running the test-set builder using the naive topology_var_map (copied from Phase B's
INTERVENTION-divergence pairs) produced **zero** usable questions in all 4 topologies.
Root cause, verified directly: association/intervention divergence for the (treat_var,
target_var) pairs the model was ACTUALLY TRAINED ON in Phase A (`causal_dag_task.QA_CONFIG`)
is ~0.01 for chain, ~0.01 for fork, ~0.00 for collider, and 0.41 for confounded. This is not
a bug -- it is the direct, intended consequence of Phase A's Step 1.3 design, which
deliberately built chain/fork/collider as *unconfounded control cases* and confounded as
the *one* topology with a real backdoor path. Testing Phase C's directional-bias prediction
requires a topology where the two true answers actually differ; only confounded has that
property for in-distribution questions. Using a different, out-of-distribution variable
pair for chain/fork/collider (e.g. forcing a collider's C variable, never used as a `do`
target during Phase A training) would confound any observed "errors" with general
out-of-distribution incompetence rather than isolating the ablation's effect -- so that
option is rejected.

**Scope narrowing, decided now:** the interventional test set (Prediction 1) is restricted
to the **confounded topology only**, with n increased from 25 to 100 to compensate for
using one topology instead of four. Predictions 2 and 3's associational test set is
similarly built from confounded-topology association questions, count-matched. All three
predictions' thresholds (0.65 effect size, 10pp tolerance, 2x asymmetry) are unchanged from
above. This narrows what Phase C can say about *cross-topology* generalization of the
ablation signature (it can no longer claim the directional bias holds on chain/fork/collider
specifically, since those topologies cannot produce a real test of it), but the core
directional-bias claim itself is still fully testable and still meaningful.

### Result (fill in AFTER running the experiment — do not edit the hypothesis above once the corresponding phase begins)

**All three predictions NOT SUPPORTED, but with an important finding underneath.** Full
write-up in `results/phase_c_conclusion.md`; summary:

- P1 (directional bias): exactly 50/50 split, p=0.540. Not supported.
- P2 (selectivity): associational accuracy dropped 98%->79% (19pp, exceeds 10pp tolerance).
  Not supported.
- P3 (asymmetry): interventional accuracy was 0% both before AND after ablation (floor
  effect -- no drop to measure). Not supported.
- **The real finding, discovered by inspecting raw predictions rather than trusting the
  pass/fail summary:** the UN-ablated model was already at 0% interventional accuracy on
  confounded-topology questions, answering in the associational direction 100% of the time
  before any ablation -- Pearl's predicted failure mode was already the baseline, leaving
  no room for Prediction 1 to detect a *shift*. Ablation didn't push further toward
  association; it collapsed the model to a single constant output (0.10) for all 100
  interventional questions, while associational-question outputs remained bimodal but
  recalibrated (0.85->1.0). This is genuine information about the circuit (it maintains
  input-sensitivity, not correct do-calculus) but not what P1-P3 were designed to detect.
- New caveat surfaced for Phase B: its transfer numbers show mechanistic consistency, not
  correctness -- the "clean" runs it patched from may themselves have been confidently
  wrong. Flagged for any future write-up.

---

## Phase D — RQ4: Behavioral Correlate

**Hypothesis:** The accuracy gap between implicitly-phrased and explicitly
do-calculus-scaffolded interventional prompts narrows at the same training step as the
RQ1/RQ2/RQ3 signals.

**Falsification condition:** The implicit/explicit gap closes at a training step
uncorrelated with (or absent at) the RQ1 LLC transition and RQ2/RQ3 circuit signals.

### Result (fill in AFTER running the experiment — do not edit the hypothesis above once the corresponding phase begins)

**Effectively falsified, but for an informative reason: no meaningful gap existed to
close.** Full write-up in `results/phase_d_conclusion.md`; summary:

- Evaluated matched implicit/explicit prompt pairs (in-distribution variable pairs, per the
  Phase C lesson) across all 10 Phase A checkpoints, n=15/topology, with Wilson-score CIs.
- Of 40 (checkpoint x topology) cells, only 1 (chain, iteration 90) showed a
  statistically real gap -- and it was a large NEGATIVE gap (explicit scaffolding drove
  accuracy from 53% to 0%), not the hypothesized positive one. All other apparent gaps,
  some as large as 33-53pp in raw terms, had overlapping confidence intervals (n=15 is
  small) and are not distinguishable from noise.
- Confounded topology -- the only topology where a real gap would be diagnostic of
  intervention-vs-association reasoning -- showed exactly 0% accuracy for BOTH implicit
  and explicit prompts (including a variant with explicit do-operator notation) at all 10
  checkpoints. This is a floor, not parity from competence.
- This independently corroborates Phase C's central finding via a completely different,
  purely behavioral method (no ablation, only prompt variation): the model has no
  confounded-topology interventional competence for any prompt style to surface, at any
  point in training.
- Disclosed confound: explicit prompts average ~1.8x longer than implicit (245 vs. 137
  chars); length was not controlled for.
- Cross-phase integration figure: `results/cross_phase_integration.png`.

**Follow-up analyses (after the user asked how to strengthen this result):**
- Length confound resolved: a length-matched, content-free `neutral_control` condition was
  identical to `implicit` at all 10 checkpoints, including the two where `explicit`
  diverged -- confirming both real effects were about scaffolding content, not length.
- Chain/fork/collider's high accuracy verified as genuine (not memorized constants): a
  discrimination test (n=40/topology, checkpoint 135) found 0.985-0.998 correlation between
  predicted and true answers, with correct directional sensitivity to `do_value` in all
  three topologies -- real, if coarse/binary-ish, causal competence. This sharpens the
  picture: the model has genuine competence on unconfounded topologies and a complete
  floor specifically on confounded, not uniform incompetence everywhere.

---

## Phase E — RQ5: Robustness/Collapse

**Hypothesis:** Recursive RL training on the model's own self-generated correct rollouts
degrades interventional/counterfactual competence (and the RQ3 ablation signature) faster
than associational competence.

**Falsification condition:** Interventional/counterfactual accuracy and the RQ3 signature
degrade at the same rate as (or slower than) associational accuracy under recursive
self-training.

### Pre-registered predictions and design (written BEFORE any Phase E experiment code exists)

**Adapting to what Phases C/D actually established, disclosed up front:** the original
hypothesis assumes a clean baseline "RQ3 ablation signature" (a directional
associational-bias fraction) to track degradation of. Phase C found this signature did NOT
exist at baseline on confounded topology (the only topology with real association/
intervention divergence) -- ablation there collapsed the model to a constant rather than
producing a directional shift, and confounded's un-ablated interventional accuracy was
already 0%. There is no clean signature to "degrade further." The predictions below are
adapted accordingly, using what Phase D additionally established: genuine, verified,
direction-correct causal competence exists on chain/fork/collider (r=0.985-0.998,
discrimination test), while confounded is a pre-existing floor. Phase E tracks the former
(real competence that COULD degrade) and separately monitors the latter (to see if a floor
can get "more confidently wrong" without changing its 0% accuracy).

**Design correction to the starter pseudocode, disclosed:** the starter code's Conditions A
and D pseudocode both call identical rft_iteration() on identical fresh-oracle batches
every iteration -- as literally written, they would not differ mechanistically at all,
defeating the entire point of isolating "recursion" as the causal factor. Corrected design:
- **Condition A (vanilla recursive):** generation g's ENTIRE training corpus is a FIXED
  snapshot of generation (g-1)'s own oracle-filtered-correct rollouts on a batch of
  questions generated ONCE per generation (not resampled each fine-tuning step). No fresh
  oracle questions enter after the initial batch is drawn.
- **Condition B (real-data-anchored):** same as A, but ~7.5% of each generation's training
  corpus is freshly oracle-labeled examples (real target answers, not model-generated).
- **Condition C (diversity-filtered):** same as A, but caps accepted rollouts per
  (topology, rung) bucket before fine-tuning, to prevent the corpus from being dominated by
  whichever bucket the model finds easiest to generate correct answers for.
- **Condition D (no-recursion control):** continues Phase A's ORIGINAL training loop
  exactly -- fresh, newly-sampled oracle question batches every single iteration, same as
  `phase_a_main.py` -- for the same number of gradient updates as condition A's
  corresponding generation used. The model's own generations are still filtered for
  correctness (as in Phase A), but the *question pool* never stops refreshing with new
  oracle-generated content, so there is no generation-to-generation self-referential corpus.

**Scale:** starting from the iteration-135 checkpoint. 3 generations per condition (not 4,
given compute constraints already observed in Phases A-D on this hardware).

**Prediction 1 (adapted from the original "circuit vs. raw accuracy" claim).** In Condition
A, chain/fork/collider's discrimination-test correlation (predicted vs. true answer,
Phase D's method) degrades faster across generations 0->3 than Condition D's does. "Faster"
means: Condition A's correlation drop (gen 0 to gen 3) is at least 2x Condition D's drop,
with non-overlapping bootstrap CIs on the gen-3 values.

**Prediction 2 (Seddik et al. replication).** Condition B's discrimination-correlation drop
(gen 0 to gen 3) is smaller than Condition A's -- real-data anchoring measurably slows
degradation relative to vanilla recursion.

**Prediction 3 (standard collapse replication).** Output diversity (std. dev. of the
model's generated numeric answer across 10 repeated samples of the same prompt, temperature
0.8) shrinks in Condition A over generations, and shrinks less in Conditions B and D.

**Also tracked, not pre-registered as a pass/fail prediction (since no clean baseline
exists to test degradation against):** whether confounded-topology's ablation-under-mean-
ablation behavior (Phase C's constant-collapse pattern) changes qualitatively across
generations in Condition A -- e.g. collapses to a different constant, or the
associational-question recalibration effect (0.85->1.0 in Phase C) shifts further.

**Interpretation table:**
- All three predictions supported -> recursive training specifically damages verified
  causal competence faster than continued non-recursive training, and real-data anchoring
  mitigates it -- the clean, publishable Pattern 1 result.
- P1 fails, but Condition A still degrades no faster than D -> recursion isn't the
  differential factor; further training in general drifts the model (Pattern 4 in the
  original spec).
- No degradation in any condition over 3 generations -> either genuinely robust at this
  scale/horizon, or 3 generations is too short to see an effect (Pattern 3) -- would need
  more generations to distinguish, noted as a limitation rather than extended given time
  constraints already flagged for this phase.

This pre-registration is committed and pushed to public GitHub BEFORE any Phase E
experiment code is written, per the same discipline used in Phase C.

### Result (fill in AFTER running the experiment — do not edit the hypothesis above once the corresponding phase begins)

**Not supported, in the opposite direction than predicted -- a real and surprising
finding.** Full write-up in `results/phase_e_conclusion.md`; summary:

- Conditions A, B, C (all recursive-training variants) showed **zero measurable
  degradation** in genuine causal discrimination (chain/fork/collider) across all 4
  generations -- correlations identical to 10+ decimal places from generation 0 to 3,
  confirmed as genuine (not a coincidental constant) both mathematically (a constant can't
  produce r=0.99 with varying true answers) and via direct token-level diagnostic
  (different questions get different, correct-direction answers).
- Condition D -- the non-recursive control, expected to be the stable baseline -- collapsed
  instead: undefined correlation (constant output) by generation 1, correlation flipped to
  -0.987 (answering backwards) by generation 2, with the ablation check producing zero
  parseable outputs. Confirmed via direct diagnostic: D-gen-2 answers "10%" to every
  question regardless of content.
- P1, P2 not supported (recursion wasn't the damaging factor; D was). P3 inconclusive
  (diversity metric underpowered as implemented).
- Confounded-topology floor (0% accuracy) persisted unchanged in every condition and
  generation -- neither harmed nor fixed by any training regime tested.
- **Important disclosed confound:** Condition D's implementation differs from A/B/C in
  optimization dynamics (many small-batch immediate updates vs. few large-batch epochs),
  not just data source -- so this result shows something real and asymmetric happened, but
  cannot cleanly isolate "recursion" as the cause vs. the accompanying difference in
  training regime. Flagged as the natural next follow-up.
- Condition D generation 3 was not completed (Colab session limits after an earlier
  runtime reset); trend from gen 0->2 was clear enough for the headline finding but the
  4-generation comparison is incomplete for D specifically.
