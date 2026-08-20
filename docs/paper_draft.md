# Paper Draft (Markdown — will convert to TMLR LaTeX template once content is stable)

Status: Results section drafted. Introduction, Related Work, Method, Discussion,
Limitations, Conclusion, Abstract still to come — see `docs/paper_outline.md` for the
full skeleton and evidence-source mapping.

---

## 5. Results

### 5.1 A real, localized, transferable circuit exists

Activation patching over every attention head and MLP layer in GPT-2 small, on a
post-RFT checkpoint verified to perform above chance on interventional questions
(55% accuracy, n=20), identified a small, sparse set of components responsible for
interventional reasoning: attention head L7H5 dominates, with a normalized patching
effect of 0.505 — more than 3x the next-largest contributor (L10H7, 0.150; L8H11,
0.123) — alongside MLP layers 7, 9, 10, and 11 (effects 0.106–0.374). All components in
layers 0–6 show approximately zero effect. This is a sparse result: 6 of 156 measured
components (144 attention heads + 12 MLP layers) account for nearly the entire measured
effect, in contrast to the diffuse, many-component pattern that would suggest no
genuine localization.

This circuit generalizes across topologies with high fidelity. Patching only these six
components — using clean-run activations to restore corrupted-run behavior — on fresh,
held-out question pairs restores 90.4–98.1% of clean-run output across all four
topologies (chain 0.981, fork 0.974, collider 0.904, confounded 0.974), including
collider (which involves the qualitatively distinct explaining-away phenomenon) and
confounded (which involves genuine backdoor confounding). A random-circuit baseline —
25 randomly sampled circuits of the same size and composition (3 attention heads + 4
MLP layers), scored identically — restored a mean of only 26.7% (std 17.4%, range −0.5%
to 53.4%) of clean-run behavior. The identified circuit outperformed all 25 random
circuits (100th percentile), ruling out the alternative explanation that any small
set of late-layer components would achieve similar restoration simply because the
task's single-token, rigidly templated output format leaves little room for the model
to express its answer differently.

This finding replicates, and in some respects strengthens, at a second scale and
architecture. Applying the identical methodology to an independently trained
Pythia-410M model (24 transformer layers, 16 attention heads, GPT-NeoX architecture —
structurally unrelated to GPT-2's implementation) identifies an even more concentrated
circuit: a single attention head, L13H10, accounts for a normalized effect of 0.614 —
approximately 9x the next-largest contributor (L17H12, 0.069) — alongside MLP layers
21, 18, 19, and 16. Cross-topology transfer is higher-fidelity than in GPT-2 (96.8–98.8%
across all four topologies, versus GPT-2's 90.4–98.1%), and the random-circuit baseline
separation is larger (candidate circuit: 98.2% restoration; 25 random circuits: mean
9.7%, std 8.2%, max 40.7% — again a clean 100th-percentile separation).

*[Figure: circuit heatmaps, GPT-2 vs. Pythia-410M side by side —
`results/phase_b_head_heatmap.png`, `results/phase_b_pythia_head_heatmap.png`]*

### 5.2 Mechanistic consistency without behavioral correctness

The existence of a real, general, validated circuit does not imply that circuit
computes the correct answer. Mean-ablating the six-component GPT-2 circuit and
evaluating on 100 confounded-topology interventional questions (paired with 100
count-matched associational questions) reveals that the un-ablated model was already
at floor: 0% interventional accuracy before any ablation was applied. Inspecting raw
outputs shows why — the un-ablated model answers every confounded-topology
interventional question with one of exactly two values (0.85 or 0.10), tracking the
*associational* direction of the underlying variables with apparent perfect fidelity,
never approaching the true interventional answer (≈0.50 in every case, since
intervening on the treatment variable severs it from the confound entirely, collapsing
the target's distribution to its unconditional marginal). This is precisely the
associational-substitution failure mode Pearl's theory predicts as the naive
alternative to genuine do-calculus reasoning — present as the model's baseline
competence, not something ablation reveals.

Ablation itself does not push the model further toward the associational answer, as
the pre-registered directional-bias hypothesis predicted; instead it collapses the
model's output to a single constant (0.10 for all 100 interventional questions,
regardless of the true associational direction). The resulting classification —
whether each ablated answer lands closer to the true interventional or true
associational value — splits at exactly 50.0%/50.0% (binomial test p = 0.540, not
significant), a geometric artifact of the collapse rather than genuine unbiased error:
by coincidence of where the true answers fall relative to the fixed output value 0.10,
roughly half the questions score as "closer to associational" and half as "closer to
interventional." On the paired associational test set, ablation causes a milder,
selective degradation (98.0% → 79.0% accuracy, a 19-percentage-point drop) — the model
remains bimodal (discriminating high vs. low) but recalibrates one of its two output
values, producing accuracy loss without full collapse.

This precise pattern — 0% un-ablated interventional accuracy via associational
substitution; an exact 50.0%/p=0.540 directional split under ablation; ablation causing
output collapse rather than directional correction — replicates almost exactly on
Pythia-410M, independently trained with a different architecture. The directional-bias
result matches to the decimal: 50.0% closer-to-associational, p = 0.540, in both
models. One difference is notable: while GPT-2's ablation collapses only interventional
output to a constant (associational output remains discriminative, explaining its
milder 19-percentage-point drop), Pythia's ablation collapses **both** question types to
a single constant (0.85), producing a substantially larger associational accuracy drop
(98.0% → 43.0%, 55 percentage points). We report this divergence honestly as a
scale-dependent difference in the *severity* of the collapse — the underlying
qualitative phenomenon (a real, localized, general circuit computing a consistent,
verified-non-random, but behaviorally incorrect result on the one topology requiring
genuine causal reasoning) is what replicates, not the exact magnitude of every
downstream effect.

*[Table: GPT-2 vs. Pythia-410M ablation results side by side, from
`results/phase_c_pythia_conclusion.md`]*

### 5.3 Genuine but coarse causal discrimination on unconfounded topologies

The confounded-topology floor does not indicate the model has learned nothing real.
On chain, fork, and collider — topologies where, by construction, association and
intervention coincide, so no genuine do-calculus distinction is required — the model's
predictions correlate strongly with true, continuously-varying interventional answers
(chain: r = 0.985; fork: r = 0.989; collider: r = 0.998, GPT-2), and correctly shift
direction between do_value = True and do_value = False groups in all three cases. This
rules out the alternative explanation that apparent competence on these topologies
reflects a memorized near-constant landing within tolerance by chance: a genuinely
constant-output model cannot produce correlation near 1.0 with continuously varying
true answers. Pythia-410M shows statistically indistinguishable correlations (chain:
0.985; fork: 0.989; collider: 0.998), again with correct directional discrimination in
all three topologies.

One caveat is worth stating precisely rather than glossing as full continuous
probability estimation: predictions are not continuous. Within each topology and
do_value group, the model outputs exactly one of two clustered values (zero
within-group variance in both GPT-2 and Pythia) rather than graded estimates matching
the finer variation in the true answers. The honest characterization is that the model
has learned a genuine, input-sensitive **binary discrimination** of the intervention's
direction on unconfounded topologies — real, replicated competence, but coarser than
full causal inference.

### 5.4 Three independent attempts to close the gap, each failing for a distinct, diagnosed reason

Given that a real circuit exists and genuine (if coarse) competence is demonstrable on
unconfounded topologies, we tested three categorically different interventions aimed at
surfacing or installing genuine confounded-topology interventional competence.

**Ablation** (§5.2) rules out one candidate explanation directly: the un-ablated
model's floor-level performance shows there is no hidden, ablation-suppressed
competence to reveal. This was not designed as a fix, but its result is informative —
any account of the confounded-topology failure must explain a genuine baseline
incapacity, not a masked capability.

**Inference-time do-calculus scaffolding** — providing explicit intervention framing in
the prompt itself, using three independently-worded variants that name the
do-operator, instruct the model to ignore normal causes, and (in one variant) write out
`P(target | do(var=val))` notation directly — never improved confounded-topology
accuracy above 0%, at any of ten training checkpoints spanning the entire RFT run. A
length-matched neutral-filler control ruled out prompt length as a confound: accuracy
under the filler condition was identical to the unscaffolded condition at every
checkpoint, including the two checkpoints where the true scaffolded condition showed a
(non-floor, non-confounded-topology) effect.

**Training-time do-calculus scaffolding** — baking the same kind of explicit reasoning
into RFT training targets themselves, so the model must learn to produce and act on the
content rather than receiving it externally at each evaluation — also failed, for a
different, precisely diagnosed reason. Two matched conditions (one warm-started with
100 intervention examples carrying a worked do-calculus rationale before the answer,
one identical but without the rationale) were trained through the full 150-iteration
RFT curriculum. Both hit the same 0% confounded-topology floor at the final checkpoint;
inspecting raw generations showed neither condition produces any rationale content by
that point — both collapse to a bare percentage answer, since RFT's acceptance
criterion rewards only the final parsed answer and has no mechanism to preserve
intermediate reasoning content. A follow-up diagnostic, evaluating the *pre-RFT*
warm-up checkpoint directly (eliminating RFT's erosion as a factor), sampled 400
completions (4 per question, matching RFT's own sampling temperature, across 100
confounded-topology questions) and found the rationale was essentially never produced
in the first place — 0 of 400 completions contained it. We trace this to warm-up
dosage: only approximately 6% of the warm-up training set consisted of
confounded-topology rationale examples (100 intervention-rationale examples split
across four topologies, diluted further by 300 plain association examples), likely
insufficient to establish the rationale as a reliable generation mode for this specific
topology before RFT ever had the chance to reinforce or erode it.

We report this section's contribution as the precise diagnosis, not merely the failure:
three structurally distinct reasons (no hidden competence to reveal; scaffolding
provided but not usable; scaffolding attempted but never robustly established) rule out
three different classes of low-effort intervention, narrowing the space of what a
genuine fix would require — likely substantially heavier, confounded-topology-specific
training exposure, evaluated in isolation from the dilution and erosion effects
documented here.

*[Table: summary of all three interventions, their result, and diagnosed reason —
new figure/table needed]*

### 5.5 Emergence timing and the Local Learning Coefficient

We separately investigated whether Pearl's predicted staged-emergence order
(association before intervention before counterfactual) is detectable in training
dynamics, and whether the Local Learning Coefficient (LLC) — a loss-landscape
complexity measure previously linked to phase transitions in other settings — tracks
any such transition. An initial run's association-rung warm-up (400 examples, 3
epochs) saturated association accuracy to ceiling before RFT began, making its
emergence structurally untestable; intervention and counterfactual showed weak,
noisy jumps (iterations 10 and 86 respectively, via a pre-specified rolling-mean
detection criterion) in the wrong predicted order. The LLC trajectory showed one
clear, twice-replicated early transition (iterations 10–30) that aligned with neither
rung-specific jump.

A follow-up run addressed the warm-up confound directly: reducing warm-up to 60
examples and 1 epoch left association at 23–27% accuracy (format-competent but with
genuine headroom), and doubling the per-iteration batch halved accuracy-quantization
noise. This produced a materially cleaner behavioral result — association now shows a
real, isolated, detected jump at iteration 81 (dipping to ≈0.30–0.40 through iteration
70, rising to a sustained ≈0.55–0.70 thereafter), consistent with Pearl's predicted
ordering. The LLC trajectory, however, was flat within noise across all 15 checkpoints
(19.3 → 18.1–18.4, per-checkpoint std 1.4–1.9 comparable to the entire observed range)
— a *weaker* LLC-behavior correspondence than the original run, with no detectable
transition anywhere, including at the newly-isolated iteration-81 jump. Across two
independently designed experiments, the LLC signal has never once aligned with a
specific, behaviorally-verified rung transition — a genuine negative result about this
particular tool's sensitivity to this particular kind of capability change, reported
as a supporting sub-study rather than a load-bearing claim.

### 5.6 Robustness under recursive self-training

We tested whether recursively fine-tuning the model on its own self-generated,
self-accepted outputs (as opposed to continued training on fresh oracle-generated
data) degrades genuine causal competence faster, following the concern that recursive
self-training can compound errors across generations. Contrary to the pre-registered
prediction, three recursive conditions showed **zero measurable degradation** across
four generations — the unconfounded-topology discrimination correlation (§5.3's
metric) was identical to more than ten decimal places between generation 0 and
generation 3 in all three conditions. The non-recursive control, expected to be the
stable baseline, instead collapsed: by generation 1 two topologies' correlations
became undefined (zero-variance, constant output), and by generation 2 the model
answered every question identically regardless of content — confirmed by direct
token-level inspection, ruling out both an evaluation bug (each checkpoint is loaded
fresh from disk) and a lucky-constant coincidence (a constant-output model cannot
produce correlation near 1.0 with genuinely varying true answers, which is what the
stable conditions showed).

This striking asymmetry is disclosed with an important confound: the collapsing
condition differed from the stable conditions not only in data source (fresh vs.
self-generated) but also in optimization dynamics — many small-batch, immediate-update
iterations (43+ separate fine-tuning calls per generation) versus fewer, larger,
epoch-structured updates over a fixed corpus. A follow-up 2×2 design crossed recursion
and optimization-dynamics independently; all four resulting conditions were stable,
including the cell using the exact mechanism (fresh data, small-batch updates) that
had collapsed originally. Auditing gradient-step counts revealed why this comparison
is not yet conclusive: the new small-batch conditions received roughly 4x fewer total
gradient steps per generation (63–69) than the original collapsing condition (288,
which was deliberately step-matched to the stable conditions by sampling repeatedly
until a target was hit) — introducing training volume as a third, unintended
confounded variable. We report the honest current state: training volume is now a
plausible third candidate explanation for the original collapse, alongside recursion
and batch dynamics, and the fully clean next design (holding total gradient steps
constant while independently varying recursion and update structure) is specified but
not yet run.

Independent of this open question, the confounded-topology floor (§5.2) was unchanged
in every condition and generation tested, including the one that collapsed
catastrophically on unconfounded topologies — there was no competence there to begin
with, and none of the training manipulations tested changed that either direction.
