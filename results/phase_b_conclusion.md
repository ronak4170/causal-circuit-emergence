# Phase B Conclusion: RQ2 (Structural Generalization)

**Model:** GPT-2 small, Phase A checkpoint at iteration 135 (verified at 55%
fresh intervention accuracy, n=20, well above chance -- see
`src/phase_b_setup.py` output).

## Method deviation, disclosed up front

The starter design assumes the candidate circuit is identified separately
*per source topology*, then tested via a full source x target transfer
matrix. Step 3 here instead identified **one** candidate circuit from a
single mixed batch spanning all 4 topologies at once (60 pairs, 15/topology),
since that's how the patching sweep was actually run. Step 4 was adapted
accordingly: rather than a 4x3 source->target matrix, the single mixed-batch
circuit's effect was measured separately on each topology's own **fresh**
held-out pairs (new random seed, not reused from Step 3). This tests
topology-generality directly, but with a caveat: each topology's own data
contributed to the original (mixed) circuit identification, so this is not
as strictly "held-out" a test as training on 3 topologies and testing purely
on a 4th untouched one would be. That stronger version is listed under
follow-ups below.

## Candidate circuit (Step 3)

From patching all 144 attention heads + 12 MLP layers individually, averaged
over 60 pairs: **L7H5** dominates (normalized effect 0.505, >3x the next
head), with **L10H7** (0.150) and **L8H11** (0.123) as secondary heads.
**MLP layers 7, 9, 10, 11** show the next tier of effects (0.106-0.374),
while all of layers 0-6 are ~0. Full breakdown in
`results/phase_b_candidate_circuit.md`. This is a small, sparse result (6 of
156 components account for nearly all the measured effect) -- not the
diffuse, dozens-of-heads pattern the starter guide flagged as a possible
failure mode.

## Cross-topology transfer result (Step 4)

Patching **only** these 7 components (using clean-run activations from that
same topology, corrupted-run tokens as the target) on **fresh** held-out
pairs:

| Topology | Mean normalized restoration (n=15) |
|----------|-------------------------------------|
| chain | 0.981 |
| fork | 0.974 |
| collider | 0.904 |
| confounded | 0.974 |

(0 = unpatched corrupt baseline, 1 = full clean-run restoration.)

## Judgment: full transfer supported, with one important caveat

Using the pre-specified categories: this is **closest to "full transfer"**
-- all four topologies restore to 90-98% of the clean-run answer using the
same tiny 7-component circuit, including collider (the topology the starter
guide specifically flagged as qualitatively different, involving
explaining-away) and confounded (the topology with genuine backdoor
confounding). There is no meaningful gap between "structurally similar"
topologies (chain/fork, both ~0.97-0.98) and the more different ones
(collider 0.90, confounded 0.97) -- if anything collider is the *lowest*,
but the gap to the others is small and could plausibly be an artifact of
collider intervention questions being intrinsically harder for the model in
general (an alternative explanation flagged explicitly in the starter
guide's Step 6) rather than evidence of weaker circuit transfer specifically.
This project did not separately measure the model's un-patched, non-circuit
accuracy per topology to fully rule out that confound -- see follow-ups.

**The important caveat:** because the circuit was identified from a mixed
batch that already included each topology's own data, this result shows the
circuit *generalizes to new questions within topologies it was partly
trained on*, which is weaker than showing it transfers to a topology it
never saw examples from at all. Given how strong and uniform the transfer is
(no topology anywhere near baseline), the more likely interpretation is that
this model learned a genuinely shared, topology-general "compute the
intervention's effect" mechanism concentrated in L7H5 + a handful of
late-layer MLPs -- but a stricter leave-one-topology-out version of this
test (below) would be needed to fully rule out the weaker alternative.

## Random-circuit baseline check (added after initial write-up)

A natural objection to the transfer result above: this task is narrow and
rigidly templated (a single percentage token, read at one teacher-forced
position), so perhaps *any* small set of late-layer components would
restore most of the clean-run answer, simply because the task gives the
model so little room to express its answer differently. If true, "we found
THE circuit" would be a much weaker claim than the transfer numbers alone
suggest.

Tested directly: 25 random circuits of the same size and composition (3
random attention heads + 4 random MLP layers, uniformly sampled) were scored
with the identical cross-topology procedure, on the identical held-out pairs
(`src/phase_b_random_baseline.py`).

**Result:** random circuits restored a mean of only **26.7%** (std 17.4%,
range -0.5% to 53.4%) of clean-run behavior, vs. the candidate circuit's
**95.8%**. The candidate circuit outperformed **all 25** random circuits
(100th percentile) -- see `results/phase_b_random_baseline_hist.png` for the
full distribution. This rules out the "any late-layer components would do
this" alternative explanation: L7H5 and its supporting components are doing
something the vast majority of other components (including other late-layer
ones, since layers were sampled from all 12) are not. This substantially
strengthens confidence in the Step 3/4 findings above.

## Follow-ups for a stricter version of this test

1. **Leave-one-topology-out circuit identification**: repeat Step 3 using
   only 3 of the 4 topologies, then test transfer to the held-out 4th
   topology never seen during circuit identification. This is the version
   that would most cleanly settle the mixed-batch caveat above. (The
   random-circuit check above addresses a different concern -- task
   narrowness -- and does not substitute for this.)
2. **Per-topology baseline accuracy**: measure the model's own (unpatched)
   generation accuracy separately per topology, to rule out "collider is
   just intrinsically harder" as an alternative explanation for its
   slightly lower transfer score.
3. **Path patching** on L7H5 specifically, to trace which upstream
   components feed it and which downstream components it feeds -- Step 3's
   plain activation patching identifies *that* this head matters, not *how*
   it computes the intervention effect.

## Checkpoint / circuit carried into Phase C

Phase C's ablation-signature testing (per Step 5's note that B and C can run
partially in parallel) should use this same candidate circuit -- **L7H5,
L10H7, L8H11 heads; MLP layers 7, 9, 10, 11** -- on the same iteration-135
checkpoint.
