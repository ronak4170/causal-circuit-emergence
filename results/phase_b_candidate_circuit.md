# Phase B Candidate Circuit (Step 3.4)

**Model:** GPT-2 small, Phase A checkpoint at iteration 135 (final checkpoint; see
`results/phase_a_conclusion.md` for why this checkpoint was selected over an
unambiguous "post-transition" point that didn't exist).

**Method:** activation patching (single-token logit-diff metric, normalized
0 = matches unpatched corrupt run, 1 = matches clean run) over 60
clean/corrupt intervention-question pairs spanning all 4 topologies (15
each), patching every attention head's `hook_z` and every layer's
`hook_mlp_out` individually. Full results: `results/phase_b_patching_results.pkl`,
`results/phase_b_head_heatmap.png`, `results/phase_b_mlp_barplot.png`.

## Candidate circuit

**Attention heads** (threshold: normalized effect > 0.10 — chosen post hoc after
seeing a large gap between the top 3 heads and everything else, not
pre-registered):

| Head | Effect |
|------|--------|
| L7H5 | **0.505** |
| L10H7 | 0.150 |
| L8H11 | 0.123 |

L7H5 dominates: its effect is more than 3x the next-largest head, and every
other head/layer combination (141 of 144 heads) falls below 0.05. This is a
notably clean, sparse result — not the "dozens of heads all show some
effect" failure mode the starter guide warned about.

**MLP layers** (same 0.10 threshold):

| Layer | Effect |
|-------|--------|
| L10 | **0.374** |
| L9 | 0.180 |
| L11 | 0.177 |
| L7 | 0.106 |

MLP effects are ~0 through layer 6, then rise sharply starting at layer 7,
peaking at layer 10. This matches the starter guide's expectation that MLPs
do more "combining information" work for this kind of reasoning than pure
attention-based name-copying (as in IOI) would.

## The falsifiable claim being tested in Step 4

This circuit — **{L7H5, L10H7, L8H11} attention heads + {L7, L9, L10, L11}
MLP layers** — is claimed to be causally responsible for this model's
interventional-reasoning competence, at least on the topologies it was
identified on. Step 4 tests whether patching *only* these components,
using activations from one topology's clean run, restores correct behavior
on a *different* topology's corrupted run — i.e., whether this is a
topology-general "compute the effect of an intervention" mechanism or a
topology-specific pattern-matcher.

## Caveat on threshold selection

The 0.10 cutoff was chosen after looking at the actual distribution of
effects (there's a natural gap: the 6 selected components are all >0.10,
the next-highest is L7H7 at 0.047) rather than fixed in advance. This is
disclosed rather than presented as a blind pre-registration, consistent
with the same honesty practice used in Phase A's jump-detection writeup.
