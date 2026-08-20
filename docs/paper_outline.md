# Paper Outline: Draft Skeleton

**Target venue:** TMLR (rolling submission, no deadline). AAAI-27's AI Alignment track is
off the table for this cycle — its abstract deadline (Aug 14, 2026) has already passed as
of this writing (Aug 19, 2026), and the full-paper deadline (Aug 21) is two days out.

**Working title (pick one, or propose alternatives):**
- "Mechanistic Consistency Without Behavioral Correctness in RL-Trained Language Models"
- "A Robust, Transferable Circuit That Computes the Wrong Answer: Causal Reasoning in
  RL-Post-Trained Language Models"
- "When the Circuit Is Real But the Answer Isn't: Causal Ladder Evaluation of RL-Trained LMs"

Every claim below is tagged with its exact evidence source so nothing gets written from
memory during drafting — always re-derive numbers from the cited file, don't trust this
outline's paraphrase of them.

---

## Abstract (sketch — write last, after all sections are drafted)

RL-post-trained language models can develop internally localized, causally-verified,
cross-topology-general circuits for a synthetic causal-reasoning task — yet that circuit
can implement a systematically wrong (associational, not interventional) answer on the one
structure that actually requires distinguishing intervention from correlation. This
mechanistic consistency without behavioral correctness replicates across two independently
trained models of different scale and architecture (GPT-2 small, Pythia-410M), and
survives three independent categories of intervention aimed at fixing it (mean ablation,
inference-time do-calculus scaffolding, training-time do-calculus scaffolding), each
failing for a distinct, precisely diagnosed reason.

## 1. Introduction

- Motivation: does RL post-training induce genuine causal reasoning (Pearl's ladder:
  association → intervention → counterfactual) or correlational shortcuts?
- Contributions (numbered list, map 1:1 to Results subsections below):
  1. A synthetic causal-DAG benchmark with an exact oracle across 4 topologies × 3 rungs.
  2. Discovery of a real, sparse, cross-topology-transferable circuit for interventional
     reasoning, validated against a random-circuit baseline.
  3. Demonstration that this circuit implements a *consistent, localized, but incorrect*
     computation on confounded structures specifically — the paper's central phenomenon.
  4. Replication of (2) and (3) at a second scale/architecture (Pythia-410M), including an
     exact-decimal match on one statistical test.
  5. Three independent, precisely-diagnosed negative results for "fixing" the gap
     (ablation, inference scaffolding, training scaffolding).
  6. Negative-but-informative results on emergence timing (RQ1/LLC) and an unexpected
     robustness finding under recursive self-training (RQ5), with disclosed confounds.

## 2. Related Work

Verified-real citations from prior research (re-verify none have since been superseded
before final submission):
- CLADDER (Jin et al. 2023) — design philosophy this benchmark adapts.
- Makelov et al. 2311.17030 — circuit faithfulness / subspace illusions, relevant to the
  "is this really the circuit" framing.
- Miller, Chughtai & Saunders 2407.08734 (COLM 2024) — ablation-methodology sensitivity;
  directly motivates the mean-vs-resample ablation robustness check (`results/robustness_followup.md`).
- Tigges et al. 2407.10827 — related interpretability methodology.
- Pythia (Biderman et al.) — the scale-up model family and its checkpoint structure.
- TMLR editorial policy (jmlr.org/tmlr/editorial-policies.html) — for the submission
  logistics section, not the science, but worth a footnote on venue choice if reviewers ask.
- Re-check `RELATED_WORK.md`'s own "last reviewed July 2026, needs re-checking" flag
  before final submission — this field moves fast (see `docs/publication_roadmap.md` item 18).

## 3. Task and Benchmark

Source: `src/causal_dag_task.py` (module docstring has the full design rationale).
- 4 topologies: chain, fork, collider, confounded — only confounded has genuine
  association/intervention divergence by construction (the others are deliberate
  unconfounded controls).
- 3 Pearl rungs per topology: association, intervention, counterfactual, each with an
  exact simulation-based oracle (rejection sampling / abduction-action-prediction).
- Single-token percentage-answer format; note the GPT-2/Pythia BPE property (every
  5%-rounded percentage 0-100 tokenizes as one token) that makes fast single-forward-pass
  scoring possible — verified for both tokenizers.

## 4. Method

- **Training**: rejection-sampling fine-tuning (RFT) as the RL post-training regime.
  Warm-up SFT (association-only) before RFT to solve reward sparsity — `src/warmup_sft.py`,
  `src/phase_a_main.py`. Same recipe reused unchanged for Pythia (`*_pythia.py` variants).
- **Circuit discovery**: activation patching over every attention head + MLP layer,
  single-forward-pass logit-diff metric (`src/phase_b_patching.py`).
- **Circuit validation**: cross-topology transfer test + random-circuit baseline
  (`src/phase_b_cross_topology.py`, `src/phase_b_random_baseline.py`).
- **Ablation methodology**: mean ablation as primary (pre-registered), resample ablation
  as a robustness check (Miller/Chughtai-motivated) — disclose the one place they disagree
  (directional-bias sub-claim) honestly.
- **Statistical practice**: pre-registration discipline (commit-before-run) for Phases C/E;
  Wilson-score CIs for small-n proportions (Phase D); multi-seed CIs for headline numbers
  (`results/robustness_followup.md`).

## 5. Results

### 5.1 A real, localized, transferable circuit exists
Source: `results/phase_b_conclusion.md`, `results/phase_c_pythia_conclusion.md` (circuit
section), `results/phase_b_pythia_*` pickles.
- GPT-2: L7H5 dominant (0.505, >3x next head) + L10H7, L8H11 + MLP 7/9/10/11. Transfer
  90-98% across topologies. Random-baseline: 100th percentile (candidate 95.8% vs. random
  mean 26.7%).
- Pythia-410M: even MORE concentrated — L13H10 alone (0.614, ~9x next head) + L17H12,
  L13H5 + MLP 21/18/19/16. Transfer 96.8-98.8%, even higher fidelity. Random-baseline:
  100th percentile (candidate 98.2% vs. random mean 9.7%).
- Figure: circuit heatmaps (`results/phase_b_head_heatmap.png`,
  `results/phase_b_pythia_head_heatmap.png`) side by side.

### 5.2 Mechanistic consistency without behavioral correctness (the central finding)
Source: `results/phase_c_conclusion.md`, `results/phase_c_pythia_conclusion.md`.
- Both models: 0% un-ablated interventional accuracy on confounded topology — the model
  substitutes the associational answer, tracking direction with apparent perfect fidelity.
- Exact-decimal match: P1 directional-bias test, 50.0% closer-to-associational, p=0.540,
  in BOTH models independently. Emphasize this as the paper's strongest single number.
- Divergence between models: GPT-2's ablation causes *selective* collapse (interventional
  only); Pythia's causes *total* collapse (both question types) — report honestly as a
  scale-dependent severity difference, not explained away.

### 5.3 Genuine but coarse causal discrimination on unconfounded topologies
Source: `results/phase_d_conclusion.md` (Follow-up 2), `results/phase_d_pythia_conclusion.md`.
- Rules out "the model learned nothing real": correlation 0.98-0.999 between predicted and
  true answers on chain/fork/collider, direction-correct, replicated at Pythia scale
  (0.985/0.989/0.998).
- Honest caveat: output is binary/discrete (two clustered values per topology), not
  continuous probability estimation — real but coarse competence.

### 5.4 Three independent, precisely diagnosed failures to close the gap
Source: `results/phase_c_conclusion.md` (ablation), `results/phase_d_conclusion.md`
(inference scaffolding), `results/phase_f_conclusion.md` (training scaffolding).
- **Ablation** (ill-suited as a "fix," included for completeness): un-ablated model was
  already at the floor; nothing to reveal.
- **Inference-time do-calculus scaffolding**: never helped, at any of 10 checkpoints,
  confounded topology 0% for implicit AND explicit prompts throughout training.
- **Training-time do-calculus scaffolding** (Phase F, the "positive fix" attempt): failed
  for a diagnosed dosage reason — thin/diluted warm-up exposure (~6% of training set was
  confounded-specific rationale content) meant the rationale never became a reliable
  generation mode, so RFT never got a chance to erode OR reinforce it. 0/400 sampled
  completions (best-of-4, matching RFT's own sampling regime) ever produced the rationale
  on confounded questions, pre-RFT.
- Frame this subsection's contribution explicitly: not "we couldn't fix it," but "we
  identified three structurally distinct reasons three different fix categories fail,"
  which is itself informative for anyone designing similar training interventions.

### 5.5 Emergence timing and the Local Learning Coefficient (RQ1) — negative, informative
Source: `results/phase_a_conclusion.md`, `results/phase_a2_conclusion.md`.
- Original run: warm-up saturated association, making its "emergence" untestable; LLC
  shows one early, replicated transition that doesn't align with any specific rung's jump.
- Redesigned run (weaker warm-up): association now shows a real, isolated jump — but LLC
  is flat across all 15 checkpoints, a WEAKER correspondence than the original run.
- Honest claim: across two independently-designed experiments, LLC never once tracked a
  behaviorally-detected rung transition — a genuine negative result about LLC's utility
  for this kind of behavioral-transition detection specifically, not about emergence
  itself.

### 5.6 Robustness under recursive self-training (RQ5) — surprising, confound disclosed
Source: `results/phase_e_conclusion.md`, `results/phase_e_2x2_conclusion.md`.
- Headline (surprising): recursive self-training conditions showed ZERO degradation across
  generations; the non-recursive control collapsed catastrophically instead.
- Disclosed confound: an optimization-dynamics difference (batch size / update frequency)
  between conditions, not just data source, confounds the interpretation.
- Follow-up 2×2 narrowed but did not resolve this — a third confound (training volume) was
  found mid-investigation. Report as open, with the precise next design specified.

## 6. Discussion

- Why might this pattern emerge? RFT's outcome-only reward has no incentive to represent
  the do-operator distinction when the associational shortcut is behaviorally
  indistinguishable during training (the model was likely never exposed to confounded
  cases where the shortcut fails during RFT's accept/reject loop, since accept/reject only
  checks final-answer correctness).
- What this says about interpretability practice generally: circuit discovery + transfer +
  random-baseline validation can all pass cleanly while the circuit computes something
  wrong — mechanistic evidence of "a real, general computation" is not evidence of
  "a correct one." Explicit warning against conflating the two, aimed at the
  interpretability community's own practice.
- Relation to Miller/Chughtai: this paper's own ablation-methodology sensitivity finding
  (mean vs. resample ablation disagreeing on the directional-bias sub-claim) is a fresh,
  concrete instance of their general warning, not just a citation.

## 7. Limitations (write this section with the same honesty as every conclusion doc)

- Scale: up to 410M parameters — informative but well below frontier scale.
- Synthetic, narrow, single-token-answer benchmark — not representative of open-ended
  causal reasoning.
- RQ5's recursion-vs-optimization-dynamics confound unresolved (precise next design
  specified but not run).
- Phase F's negative result is specific to the dosage/design tested — heavier
  confounded-specific rationale dosage in isolation remains untested future work.
- 1.4B further scale-up not attempted (assessed and deliberately deprioritized — see
  `docs/publication_roadmap.md`).

## 8. Conclusion

Restate the central finding plainly; end on the interpretability-practice warning from
§6, since that's the most broadly useful takeaway for readers who don't care about causal
DAGs specifically.

## Appendix

- Full hyperparameters (warm-up dosage, RFT settings, LR, tolerance) per experiment.
- Multi-seed CIs (`results/robustness_followup.md`).
- Ablation-methodology robustness detail (mean vs. resample, full numbers).
- Phase F's methodological bug-and-fix (greedy decoding suppressing the rationale; length
  heuristic falsely counting degenerate repeats) — worth including as a transparency note,
  since it's a genuinely useful lesson for anyone evaluating CoT/rationale-trained models
  with greedy decoding.

---

## Figures inventory (existing, reusable — check each still renders correctly before use)

- `results/phase_a_plot.png` — per-rung accuracy + LLC vs. iteration.
- `results/phase_b_head_heatmap.png` / `phase_b_pythia_head_heatmap.png` — circuit heatmaps.
- `results/phase_b_mlp_barplot.png` / `phase_b_pythia_mlp_barplot.png`.
- `results/phase_b_random_baseline_hist.png` — random-circuit baseline distribution.
- `results/phase_c_accuracy_barplot.png`.
- `results/phase_d_gap_plot.png`, `results/phase_d_discrimination_scatter.png`.
- `results/cross_phase_integration.png` — three-panel Phase A/D/LLC on shared axis.
- New figures likely needed: side-by-side GPT-2/Pythia circuit comparison; a summary
  table/figure for §5.4's three failed interventions.

## Open decisions before drafting starts

1. Confirm working title.
2. Draft in Markdown first (fast iteration) then convert to TMLR's LaTeX template, or
   start directly in LaTeX? (TMLR template needs downloading from jmlr.org/tmlr — not yet
   fetched.)
3. Which section to draft first — recommend starting with §5 (Results), since all the
   numbers and interpretation already exist in the conclusion docs and mostly need
   compression/reorganization, not new analysis. Introduction and Discussion are easier to
   write well once Results is locked.
