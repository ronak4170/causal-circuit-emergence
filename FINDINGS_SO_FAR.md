# Findings and Conclusions So Far (Phase 0, Phase A, Phase B)

**Last updated:** after Phase B completion, before Phase C.

This document synthesizes what has actually been found across the three completed
phases, separated from the phase-by-phase working notes in `docs/phase0_setup.md`,
`results/phase_a_conclusion.md`, and `results/phase_b_conclusion.md`. Read those for
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

## 4. Cross-Cutting Deductions (things learned across all three phases)

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
   signal (Phase A) is messy.** Despite Phase A's noisy, only-partially-
   supported emergence story, Phase B found an unambiguous, sparse, strongly
   transferring circuit on the *same* checkpoint. This is itself informative:
   whatever interventional competence this model has — however noisily or
   gradually it arrived — is concentrated in a small, identifiable,
   topology-general mechanism. The mechanistic finding does not depend on
   Phase A's emergence story having been clean.

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

---

## 5. What's Still Open Going Into Phase C

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
- No ablation-error-signature testing (RQ3) has been done yet — this is
  Phase C's task, using the same L7H5/L10H7/L8H11 + MLP-7/9/10/11 circuit and
  iteration-135 checkpoint identified here.
