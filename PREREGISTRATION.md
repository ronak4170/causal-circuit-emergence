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

### Result (fill in AFTER running the experiment — do not edit the hypothesis above once the corresponding phase begins)

*(empty — to be filled in after Phase C is run)*

---

## Phase D — RQ4: Behavioral Correlate

**Hypothesis:** The accuracy gap between implicitly-phrased and explicitly
do-calculus-scaffolded interventional prompts narrows at the same training step as the
RQ1/RQ2/RQ3 signals.

**Falsification condition:** The implicit/explicit gap closes at a training step
uncorrelated with (or absent at) the RQ1 LLC transition and RQ2/RQ3 circuit signals.

### Result (fill in AFTER running the experiment — do not edit the hypothesis above once the corresponding phase begins)

*(empty — to be filled in after Phase D is run)*

---

## Phase E — RQ5: Robustness/Collapse

**Hypothesis:** Recursive RL training on the model's own self-generated correct rollouts
degrades interventional/counterfactual competence (and the RQ3 ablation signature) faster
than associational competence.

**Falsification condition:** Interventional/counterfactual accuracy and the RQ3 signature
degrade at the same rate as (or slower than) associational accuracy under recursive
self-training.

### Result (fill in AFTER running the experiment — do not edit the hypothesis above once the corresponding phase begins)

*(empty — to be filled in after Phase E is run)*
