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

*(empty — to be filled in after Phase A is run)*

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
