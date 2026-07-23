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

*(empty — to be filled in after Phase B is run)*

---

## Phase C — RQ3: Ablation Signature

**Hypothesis:** Ablating the identified circuit causes the model's errors to shift
specifically toward associational-style answers (`P(Y|X)` instead of `P(Y|do(X))`) — a
theory-predicted, directional failure — rather than generic random degradation.

**Falsification condition:** Post-ablation errors are not disproportionately
associational-style (e.g. the error distribution is statistically indistinguishable from
random degradation across answer categories).

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
