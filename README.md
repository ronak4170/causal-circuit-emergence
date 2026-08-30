# Climbing the Causal Ladder: Mechanistic Evidence for the Staged Emergence, Structural Localization, and Fragility of Causal Reasoning Circuits in RL-Post-Trained Language Models

## Summary

When a language model's ability to answer causal-inference questions improves during RL
post-training, is it learning genuine causal structure — coherent, generalizable responses
to interventions (`do(X)`) and counterfactuals — or is it learning a correlational shortcut
that merely tracks surface features of associational queries and breaks down under
intervention? This project asks whether the emergence of causal reasoning capability can be
(1) detected as a distinct developmental event during training via phase-transition
signals (the Local Learning Coefficient), (2) localized to a specific, patchable circuit
inside the model, and (3) shown to be fragile — degrading in a theory-predicted direction
under ablation, and decaying faster than associational competence under recursive
self-training. The approach combines a synthetic causal-DAG benchmark spanning Pearl's
three-rung causal hierarchy (association, intervention, counterfactual) with RL
post-training, devinterp-based developmental interpretability, and TransformerLens-based
activation/path patching.

## Research Questions

Each RQ is designed to be independently falsifiable — the project can fail at any one of
these without invalidating the others.

| RQ | Question | Falsification condition |
|----|----------|--------------------------|
| RQ1 — Emergence | Do the three rungs of Pearl's causal hierarchy (association, intervention, counterfactual) emerge as separable phase transitions during RL post-training, detectable via LLC, and in that order? | No distinguishable LLC transitions align with per-rung accuracy jumps, or the three rungs emerge simultaneously / out of order. |
| RQ2 — Structural generalization | Does the circuit responsible for interventional reasoning on one causal-DAG topology (e.g. a fork) causally transfer via activation patching to a different topology (e.g. a chain)? | Patching the identified circuit's activations across topologies produces no consistent restoration of interventional accuracy — i.e. the circuit is topology-specific and does not transfer. |
| RQ3 — Ablation signature | Does ablating the identified circuit shift errors specifically toward associational-style answers (`P(Y\|X)` instead of `P(Y\|do(X))`), rather than causing generic random degradation? | Post-ablation errors are not disproportionately associational-style (e.g. error distribution is statistically indistinguishable from random degradation across answer categories). |
| RQ4 — Behavioral correlate | Does the accuracy gap between implicitly-phrased and explicitly do-calculus-scaffolded interventional prompts narrow at the same training step as the RQ1/RQ2/RQ3 signals? | The implicit/explicit gap closes at a training step uncorrelated with (or absent at) the RQ1 LLC transition and RQ2/RQ3 circuit signals. |
| RQ5 — Robustness/collapse | Does recursive RL training on the model's own self-generated correct rollouts degrade interventional/counterfactual competence (and the RQ3 ablation signature) faster than associational competence? | Interventional/counterfactual accuracy and the RQ3 signature degrade at the same rate as (or slower than) associational accuracy under recursive self-training. |

## Timeline (16 weeks)

- **Phase 0 :** Environment setup; reproduce grokking + LLC phase transition on
  modular arithmetic; reproduce the IOI circuit via activation patching on GPT-2 small.
  Validation that the tools work before touching the real research question.
- **Phase A :** RQ1 — train on a curriculum spanning all three query rungs, track
  per-rung accuracy and LLC, look for separable phase transitions.
- **Phase B :** RQ2 — identify the interventional-reasoning circuit via patching,
  test cross-topology transfer (fork ↔ chain ↔ collider).
- **Phase C :** RQ3 — pre-register the predicted ablation error pattern before
  running the experiment, then ablate and classify errors.
- **Phase D :** RQ4 — matched implicit/explicit prompt pairs, track the accuracy
  gap vs. training step, overlay on Phase A/B/C signals.
- **Phase E :** RQ5 — recursive RL generations, track degradation rates per
  query type.
- **Phase F :** Write-up, arXiv preprint, workshop submission.

## Tech Stack

- **Models:** GPT-2 small (primary), Pythia-410M (stretch goal)
- **Benchmark:** synthetic causal-DAG tasks (chain / fork / collider topologies, Pearl's
  three query rungs, ground truth via a causal-inference oracle, in the style of CLADDER)
- **Training:** RL post-training via rejection-sampling fine-tuning or lightweight GRPO
- **Interpretability:** [devinterp](https://github.com/timaeus-research/devinterp) for LLC
  estimation and phase-transition detection; [TransformerLens](https://transformerlensorg.github.io/TransformerLens/)
  for activation/path patching and circuit identification
- **Core libraries:** PyTorch, numpy, pandas, matplotlib, seaborn, scikit-learn, networkx

## Related Work

See [RELATED_WORK.md](RELATED_WORK.md) for an honest positioning of this project's
contribution against existing literature.
