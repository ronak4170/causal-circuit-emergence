# Publication Roadmap: Handoff Document

**Progress update (added after this document was first written):** this document was
handed to an external deeper-research session (claude.ai), which returned a refined
strategic plan — reframe around the "mechanistic consistency ≠ correctness" headline
(matches this doc's own §5 priority ranking), target TMLR, scale to Pythia, and a
prioritized experiment list. Several of that report's most load-bearing citations were
independently verified via web search (Makelov et al. 2311.17030, Miller/Chughtai
2407.08734, Tigges et al. 2407.10827, Lu et al. 2512.20760, TMLR editorial policy, MechRL
2605.26343, Apart Research's LLC/GRPO project, Pythia's checkpoint structure — all checked
out as real and accurately described). **The two no-new-infrastructure items from that
report's prioritized list (ablation-methodology robustness; multi-seed CIs on headline
numbers) are now done** — see `results/robustness_followup.md`. **The RQ5 2×2 (item 8
below) was also attempted**, on the same free-tier Colab GPU used throughout this project
(GPT-2 small, no new spend) — result: not fully resolved, but narrowed. All four cells of
the 2×2 were stable, including the one matching the original collapsed condition's
mechanism, but a step-count audit revealed the new run used ~4x fewer gradient steps per
generation than the original — so training volume is now a third candidate explanation
alongside recursion and batch dynamics. See `results/phase_e_2x2_conclusion.md` for the
full write-up and the precise next design (hold total gradient steps constant) needed to
finish resolving this. **The RQ1 redesign was also attempted** (item 4/6 below), again on
free-tier Colab GPU, no spend needed — result: the behavioral fix worked (association now
shows a real, isolated, detected jump, unlike the original run), but the LLC signal still
shows no detectable transition anywhere, a weaker LLC-behavior correspondence than even the
original run. See `results/phase_a2_conclusion.md`. Only the Pythia scale-up remains as a
genuine spend decision, described below.

**Purpose of this document:** a self-contained briefing for a fresh research/planning
session (no prior context on this project) to (a) understand everything found so far, and
(b) help design the follow-up work needed to make this publication-worthy at a level above
a workshop/short paper. Written after completing Phases 0 and A-E of a 16-week
undergraduate research project.

Repo: https://github.com/ronak4170/causal-circuit-emergence (all code, data, and full
phase-by-phase conclusion docs are there — this document is a compressed summary of
`FINDINGS_SO_FAR.md` plus a forward-looking research agenda that file doesn't contain).

---

## 1. The original research program

**Title:** "Climbing the Causal Ladder: Mechanistic Evidence for the Staged Emergence,
Structural Localization, and Fragility of Causal Reasoning Circuits in RL-Post-Trained
Language Models"

**Core question:** when reasoning capability emerges during RL post-training on
causal-inference tasks, does a model learn genuine causal structure (coherent responses to
interventions/counterfactuals) or a correlational shortcut (good on associational queries,
incoherent under intervention)?

**Five research questions, each independently falsifiable:**
- RQ1 (Emergence): do association, intervention, counterfactual competence emerge as
  separable, ordered phase transitions during training, detectable via the Local Learning
  Coefficient (LLC)?
- RQ2 (Structural generalization): does an intervention-reasoning circuit found on one
  causal-DAG topology transfer to a different topology via activation patching?
- RQ3 (Ablation signature): does ablating that circuit cause errors to shift specifically
  toward associational-style answers (a theory-predicted, directional failure)?
- RQ4 (Behavioral correlate): does the gap between implicitly- and explicitly-scaffolded
  intervention prompts narrow at the same training step as the internal signals?
- RQ5 (Robustness): does recursive RL training on the model's own rollouts degrade
  interventional competence faster than associational competence?

**Positioning against prior work (as of a July 2026 literature review — needs
re-verification, field moves fast):**
- CLADDER (Jin et al. 2023) / Corr2Cause (Jin et al. 2023/24): established the synthetic
  causal-DAG benchmark style. We adapted, did not invent, this.
- **arXiv:2512.20760** ("Generalization of RLVR Using Causal Reasoning as a Testbed", Dec
  2025): the closest prior work — same training regime (RLVR) and task domain (Pearl's
  ladder), but purely behavioral, no mechanistic interpretability, no LLC/phase-transition
  tracking, no circuit identification, no ablation-signature testing, no prompting-gap
  analysis. **Our intended differentiator was adding exactly that mechanistic/developmental
  layer.**
- arXiv:2604.25011 (RL + mech interp via sparse crosscoders, general math/reasoning, not
  causal-ladder specifically).
- Geiger et al. (causal abstraction formalism for why patching = causal intervention).
- Nanda et al. / devinterp program (LLC-based phase transition detection, mostly
  pretraining/grokking settings, not RL post-training).
- Wang et al. (IOI circuit — methodological template for patching).

**Model/method stack:** GPT-2 small, rejection-sampling fine-tuning (RFT) for RL
post-training, `devinterp` for LLC estimation, `TransformerLens` for activation patching.
Compute: single Apple Silicon Mac (CPU only, no local GPU) supplemented by ad hoc Google
Colab GPU sessions for the heaviest steps.

---

## 2. Complete results, phase by phase

### Phase 0 — Tooling validation (not a research result)
Reproduced grokking + LLC (2 of 3 sub-checks clean; LLC tracked training destabilization
events more than the canonical "grokking moment") and the IOI circuit on GPT-2 small (found
2 of 3 canonical Name Mover Heads via patching). Established: MPS backend avoided
throughout (TransformerLens documents it may give silently incorrect results); CPU-only
locally, Colab GPU used opportunistically.

### Phase A — RQ1 (Emergence): **partial support, weak, leaning falsification**
- GPT-2 small, warm-started via association-only SFT (to solve reward sparsity), then RFT
  across all 3 rungs × 4 topologies (chain/fork/collider/confounded), 150 iterations.
  Replicated on both local CPU and Colab GPU (different seeds).
- Association showed no detectable "emergence" because the warm-up had already saturated it
  to ~100% before RFT began — a design confound that makes RQ1's ordering claim untestable
  for that rung specifically.
- Intervention and counterfactual each showed a "jump" by a pre-specified (not strictly
  blinded) 20pp/10-iteration threshold, but both were noisy plateaus/drifts, not clean
  transitions.
- **LLC showed one clear, replicated transition** (both CPU and GPU runs, different seeds)
  in the first 15-30 iterations, then a flat plateau for the rest of training — real, but
  it doesn't align with either rung's later "jump."
- Selected checkpoint for later phases: final iteration (135) — no unambiguous
  "post-transition" point existed.

### Phase B — RQ2 (Circuit transfer): **strong support**
- Activation patching (single-token logit-diff metric, since GPT-2's BPE tokenizes every
  5%-rounded percentage as one token) across all 144 heads + 12 MLP layers, averaged over
  60 clean/corrupt intervention pairs (4 topologies).
- Found a sparse candidate circuit: **head L7H5 dominant (0.505 normalized effect, >3x next
  largest), heads L10H7/L8H11 secondary, MLP layers 7/9/10/11.**
- Cross-topology transfer: patching only these 7 components restored **90-98%** of
  clean-run behavior on fresh held-out questions across chain (0.981), fork (0.974),
  collider (0.904), confounded (0.974).
- **Random-circuit baseline check** (25 random same-size circuits): random circuits
  averaged only 26.7% restoration (std 17.4%) vs. the candidate's 95.8% — candidate beat
  all 25 (100th percentile). Rules out "any small circuit would do this on a narrow task."
- Caveat: circuit identified from a mixed-topology batch, not strict leave-one-topology-out
  — shows generalization within seen topologies, not to a wholly unseen one.

### Phase C — RQ3 (Ablation signature): **all 3 pre-registered predictions failed — but
revealed the project's most important finding**
- Pre-registered (committed to git BEFORE any experiment code existed) 3 predictions with
  exact thresholds: directional bias, selectivity, effect asymmetry.
- Test-set construction itself required a pre-registration addendum: association and
  intervention answers only genuinely diverge (for in-distribution variable pairs) on the
  **confounded** topology — chain/fork/collider were deliberately built as unconfounded
  controls in Phase A, so they can't test directional bias at all.
- Mean-ablated the Phase B circuit; **found the UN-ablated model was already at 0%
  interventional accuracy on confounded-topology questions**, answering with the
  associational shortcut 100% of the time before any ablation — leaving no room for the
  predicted "shift." Ablation didn't push further toward association; it collapsed the
  model to a single constant output (0.10) for all 100 test questions, while associational
  question accuracy dropped more mildly (98%→79%) with recalibration (0.85→1.0), not
  collapse.
- **New caveat this surfaced for Phase B:** the 90-98% transfer numbers show mechanistic
  *consistency*, not *correctness* — Phase B never checked whether the "clean" run's own
  answer was actually right.

### Phase D — RQ4 (Prompting gap): **effectively falsified, independently corroborates
Phase C — then sharpened by two follow-ups**
- Matched implicit/explicit (do-calculus-scaffolded) prompt pairs across all 10 available
  Phase A checkpoints, Wilson-score CIs (small n=15/topology).
- Only 1 of 40 (checkpoint, topology) cells showed a statistically real gap, and it was
  **negative** (scaffolding hurt: chain accuracy 53%→0% at iteration 90).
- **Confounded topology scored exactly 0% for both prompt styles at all 10 checkpoints** —
  including a variant with explicit `P(target|do(var=val))` notation. Matches Phase C's
  finding via a fully independent, non-mechanistic method.
- **Follow-up 1 (length confound):** a length-matched, content-free neutral control
  condition matched `implicit` exactly at all 10 checkpoints — ruling out prompt length as
  an explanation for the two real effects found.
- **Follow-up 2 (discrimination test):** checked whether chain/fork/collider's decent
  accuracy reflects genuine competence or a lucky/memorized constant. Found strong,
  genuine, direction-correct discrimination: **r=0.985 (chain), 0.989 (fork), 0.998
  (collider)** between predicted and true answers, with correct directional sensitivity to
  `do_value` in all three. Caveat: outputs are coarse/binary-ish (2 clustered values per
  topology), not fully graded probability estimates.
- **Net sharpened picture:** not "no causal competence anywhere" but "genuine, verified
  competence on unconfounded topologies, complete floor specifically on confounded."

### Phase E — RQ5 (Recursive robustness): **not supported, in the OPPOSITE direction than
predicted**
- Pre-registered (adapted to Phase C/D's actual floor-effect baseline) predictions;
  corrected the starter design so Conditions A (recursive) and D (non-recursive control)
  actually differ mechanistically (A/B/C: fixed self-generated corpus per generation; D:
  continues Phase A's fresh-oracle loop, matched gradient-step count).
- 3-4 generations from the iteration-135 checkpoint, run on Colab GPU (survived one full
  lost run to a Colab session reset).
- **Conditions A, B, C (all recursive variants) showed ZERO measurable degradation** in
  genuine causal discrimination across 4 generations — correlations identical to 10+
  decimal places from generation 0 to 3.
- **Condition D — the non-recursive control, expected to be stable — collapsed instead**:
  undefined correlation (constant output) by gen 1, correlation flipped to -0.987
  (answering backwards) by gen 2, ablation producing zero parseable outputs.
- Verified genuine two ways: mathematically (a constant output can't produce r≈0.99 with
  varying true answers) and via direct token-level diagnostic (A gives different, correct
  answers to different questions at generation 3; D gives the identical wrong answer to
  everything at generation 2).
- **Disclosed confound:** Condition D's implementation differs from A/B/C in optimization
  dynamics (many small-batch immediate gradient updates vs. a few large-batch epochs over a
  fixed corpus), not just data source. This is the confound flagged in the prior message —
  it means "recursion is protective" is not yet a clean, isolated claim.
- Condition D generation 3 was never completed (ran out of Colab session time after an
  earlier full run was lost to a runtime reset).

### The single most important cross-phase finding
**Mechanistic consistency (Phase B's patching transfer) is not evidence of correctness.**
A circuit can show clean, strongly-verified (beats random baseline 100th percentile),
topology-general behavior under patching while the underlying answers it's producing are
completely wrong (0% accuracy) on the one structural case that actually tests genuine
causal reasoning — discovered independently by two unrelated methods (ablation in Phase C,
prompting in Phase D).

---

## 3. The three specific weaknesses to address (from the prior assessment)

### Weakness 1: Scale
GPT-2 small; small samples throughout (15-100 examples per condition/checkpoint); CPU-only
local compute patched with ad hoc Colab GPU bursts (which caused real data loss twice —
Colab free-tier runtimes reset mid-run and wiped in-progress results, requiring
Drive-backup workarounds).

### Weakness 2: RQ1 (staged emergence) — the intended novel differentiator vs.
arXiv:2512.20760 — came back weakest
The LLC transition is real (replicated across 2 independent runs) but doesn't cleanly align
with any specific rung's behavioral transition. The association-rung warm-up design
confound makes the core "ordered emergence" claim untestable as currently designed.

### Weakness 3: RQ5's most interesting number is confounded
Condition D (non-recursive control) differs from A/B/C not just in data source (the
intended manipulation) but in optimization dynamics (batch size / update frequency). The
collapse-vs-stability finding is real and verified, but can't yet be attributed cleanly to
"recursion" as opposed to "this specific training regime's update dynamics."

---

## 4. Concrete research directions (for deeper investigation / next-session planning)

### For Weakness 1 (Scale)
1. **What's the minimum viable scale-up to be credible at a mid-tier ML venue?** Research
   what other small-model mechanistic interpretability papers (e.g. IOI, docstring circuit,
   grokking papers) considered acceptable scale, and whether GPT-2-small + larger sample
   sizes (n=100-500 instead of 15-100) would suffice, or whether a genuinely larger base
   model (Pythia-410M, already listed as this project's stretch goal) is necessary.
2. **Investigate cheaper/more reliable GPU access than free-tier Colab** — the repeated
   session resets cost real data (a full Phase E run lost once). Research: Colab Pro,
   Lambda/RunPod/Vast.ai spot instances, university compute credits, or restructuring the
   pipeline to checkpoint aggressively enough that any interruption loses <5 min of work
   (partially done for Phase A/E checkpoints, not yet for all scripts).
3. **Research what sample size / statistical power would be needed** to detect effects of
   the magnitude actually observed (e.g. Phase D's Wilson CIs needed huge gaps to be
   significant at n=15 — what n would detect a real 10-15pp gap reliably?).

### For Weakness 2 (RQ1 / staged emergence)
4. **Redesign the warm-up to avoid the association-ceiling confound.** Research options:
   (a) skip the warm-up entirely and instead solve reward sparsity with a curriculum that
   ramps up rung difficulty gradually rather than pre-training on association to 100% first,
   (b) use a much shorter/weaker warm-up (e.g., 50 examples instead of 400, or 1 epoch) so
   association starts below ceiling, (c) research how similar RLVR papers (including
   arXiv:2512.20760 itself) handle cold-start reward sparsity without this specific
   confound.
5. **Denser LLC checkpointing specifically around the intervention/counterfactual jump
   iterations** (10-30 and 75-90 in the run actually completed) to test whether the current
   15-iteration checkpoint spacing is simply too coarse to resolve real alignment that a
   finer grid would show.
6. **Larger per-iteration batch sizes** (Phase A used 4 questions/rung/iteration, producing
   accuracy quantized to {0,.25,.5,.75,1.0} — extremely noisy). Research what batch size
   would meaningfully reduce this quantization noise given the compute budget.
7. **Consider whether LLC is even the right tool for this specific claim.** Research
   whether other phase-transition-detection methods (e.g. tracking specific attention
   pattern formation directly, or the Phase B circuit's own emergence over training i.e.
   re-running Phase B's patching analysis at multiple Phase A checkpoints instead of just
   the final one) would give a cleaner signal than LLC for testing RQ1's ordering claim.

### For Weakness 3 (RQ5 confound)
8. **Design and run the controlled follow-up**: match Condition D's optimization dynamics
   exactly to A/B/C (same batch size, same epoch-over-fixed-corpus structure), varying ONLY
   whether the corpus is self-generated (recursive) or freshly oracle-sampled
   (non-recursive). This is the single most valuable next experiment to make RQ5's finding
   publication-clean.
9. **Alternatively/additionally**, run a version of Condition D using A/B/C's exact
   corpus-then-multi-epoch structure but with fresh oracle data each generation (isolating
   "big-batch-epoch-training on fresh data" as its own condition, separate from the
   original small-batch-iterative D) — this would create a proper 2×2 design (recursive vs.
   fresh data) × (small-batch-iterative vs. big-batch-epoch), fully separating the two
   confounded factors.
10. **Investigate WHY small-batch iterative training (D's original design) collapsed** —
    research whether this is a known instability mode in the RLHF/RLVR literature (e.g.
    reward hacking, catastrophic forgetting from many small updates at a fixed LR without
    warmup/decay), which would itself be a citable, useful finding independent of the
    recursion question.
11. **Complete Condition D generation 3** and consider extending all conditions to more
    generations (5-6+) if the corrected design still shows no A/B/C degradation, to check
    whether "no degradation observed" is genuine robustness or just a short observation
    window (Pattern 3 vs Pattern 1 in the original interpretation table).

### Additional directions (from the project's own "what's still open" list, worth
including in a broader research push even though not part of the original 3 weaknesses)
12. Leave-one-topology-out circuit identification for Phase B (stricter transfer test than
    the mixed-batch approach actually used).
13. Per-topology baseline (unpatched) accuracy for Phase B, to determine whether collider's
    slightly lower transfer score (0.904 vs ~0.97-0.98) reflects circuit-specificity or
    just collider questions being intrinsically harder.
14. Path patching on L7H5 specifically (Phase B identified THAT it matters, not HOW it
    computes the intervention effect) — traces upstream/downstream connections.
15. Investigate the "coarse, binary-ish output" phenomenon from Phase D's discrimination
    test — is this general to how this model produces percentage answers, or specific to
    intervention questions?
16. Phase A's counterfactual-rung jump (iteration 86) has no known mechanistic correlate —
    Phase B only investigated the intervention circuit. Does counterfactual reasoning share
    components with it?
17. Replicate the one statistically real Phase D effect (chain, iteration 90: scaffolding
    driving accuracy from 53%→0%) with a different seed to check it's not a
    single-checkpoint artifact.
18. Re-verify the literature positioning (`RELATED_WORK.md` says "last reviewed July
    2026" and explicitly flags itself as needing re-checking before any final writeup,
    given how fast this field moves) — check for newer work specifically combining RLVR +
    causal ladder + mechanistic interpretability that may have appeared since, which could
    change the novelty claim.

---

## 5. What I'd suggest a deeper-research session actually prioritize

Given limited time/compute, in priority order: **(8) the RQ5 controlled follow-up** is
probably the single highest-leverage next experiment — it's cheap (reuses all existing
Phase E infrastructure, just needs the corpus-generation mechanism unified across
conditions) and would convert an already-interesting, already-verified-genuine result into
a clean, confound-free claim. **(4/6) RQ1's warm-up redesign + larger batch size** is the
second priority since it's the piece most explicitly promised as the paper's novel
contribution over prior work. **Scale-up (1-3)** is probably the least tractable to solve
quickly and might be better addressed by explicitly framing the paper as workshop-scale
work with the mid-tier-venue scale-up as "future work," rather than trying to solve it
before writing anything up.
