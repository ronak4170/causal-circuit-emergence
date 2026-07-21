# Related Work — Honest Positioning

**Last literature review: July 2026.** This section must be re-checked before the final
writeup, as this is a fast-moving field.

This document exists to state precisely what this project adapts from prior work, what is
genuinely new, and what the closest competing work already covers — so that the eventual
write-up does not overclaim novelty.

## Benchmark design

**CLADDER** (Jin et al., 2023) and **Corr2Cause** (Jin et al., 2023/2024) already established
the synthetic causal-DAG benchmark style used here and showed that LLMs perform at
near-random-chance on pure causal inference across Pearl's ladder. This project **adapts**
their benchmark design (DAG topologies, query rungs, ground-truth generation via a causal
oracle) — it does **not** claim to invent this benchmark style.

## Training regime and task combination

**"Generalization of RLVR Using Causal Reasoning as a Testbed"** (arXiv:2512.20760,
December 2025) already trains models via RLVR specifically on Pearl's causal ladder and
studies cross-rung generalization *behaviorally*. This is the closest prior work to this
project's training regime and task combination. It does **not** include any mechanistic
interpretability, LLC/phase-transition tracking, circuit identification, ablation-signature
testing, or the implicit/explicit prompting gap. This project's contribution is explicitly
the mechanistic/developmental layer built on top of this kind of behavioral setup — **not**
the training regime or task itself, which is prior art.

## Mechanistic interpretability of RL post-training

**"Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of
Post-Training in LLMs"** (arXiv:2604.25011, April 2026) combines RL post-training with
mechanistic interpretability (sparse crosscoders, causal feature interventions), but studies
general math/reasoning generalization, **not** Pearl's causal ladder specifically. This
project narrows that same combination (RL + mech interp) onto the causal-ladder task domain.

## Theoretical foundation for causal interpretability

**Geiger et al., "Causal Abstraction: A Theoretical Foundation for Mechanistic
Interpretability"** (arXiv:2301.04709) established that activation/path patching *are* causal
interventions in the formal sense. This project builds on that framing — it does **not**
claim to have originated the idea that mechanistic interpretability is fundamentally causal.

## Developmental interpretability / LLC-based phase transitions

**Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability"** (2023),
and the devinterp/Timaeus developmental interpretability program, established LLC-based
phase-transition detection on toy tasks — mostly in pretraining/grokking settings. This
project applies that same method to RL post-training on a novel task domain (Pearl's causal
ladder), which is comparatively less studied. Phase 0 explicitly reproduces the original
grokking + LLC result before this project attempts to apply the method to a new setting.

## Circuit identification methodology

**Wang et al., "Interpretability in the Wild: A Circuit for Indirect Object Identification
in GPT-2 Small"** (arXiv:2211.00593) is the methodological template for this project's
Phase 0 activation-patching reproduction, and for the circuit-identification approach used
in Phase B (RQ2).

## Summary of this project's actual contribution

Given the above, the honest claim of novelty is: applying LLC-based developmental
interpretability and activation/path-patching circuit identification — both established
methods — to the specific combination of (a) Pearl's causal ladder as the task domain and
(b) RL post-training as the training regime, with a specific focus on three signals that, to
this literature review's knowledge, no prior work combines: staged/ordered emergence across
causal-hierarchy rungs, cross-topology circuit transfer, and a directional (associational-
leaning) ablation error signature. RQ4 (implicit/explicit prompting gap) and RQ5 (recursive
self-training fragility) are, as of this review, not addressed by the cited prior work in
this task domain.
