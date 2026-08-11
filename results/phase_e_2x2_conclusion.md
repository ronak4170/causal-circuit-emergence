# Phase E 2×2 Follow-up: Recursion vs. Optimization Dynamics vs. Training Volume

**Motivation:** `results/phase_e_conclusion.md` found that the original Phase E's
non-recursive control (Condition D) collapsed to a context-insensitive constant within 2
generations, while the recursive conditions (A/B/C) showed no degradation — the opposite of
the pre-registered prediction. That result was disclosed as confounded: Condition D
differed from A/B/C in both data source (fresh vs. self-generated) AND optimization
dynamics (many small-batch immediate updates vs. a few large-batch epochs). This follow-up
implements a proper 2×2 crossing those two factors independently: **RECURSION** (question
pool frozen and reused every generation, vs. fresh each generation) × **OPTIMIZATION**
(big-batch/3-epoch vs. small-batch/single-pass).

## Result: nothing collapsed — but a third confound was found in the process

**All four conditions (REC_BIG, REC_SMALL, FRESH_BIG, FRESH_SMALL) showed zero measurable
degradation across all 3 checkpoints (generations 0, 1, 2)** — discrimination correlations
identical to 10+ decimal places within every condition, matching the pattern the original
A/B/C showed. Critically, **FRESH_SMALL uses the exact same mechanism as the original
Condition D that collapsed** (fresh questions each generation, small-batch immediate
updates) — and here it was perfectly stable.

This looks at first like "recursion wasn't the cause, batch dynamics weren't either." But
before drawing that conclusion, the actual gradient-step counts need to be checked:

| Condition | Generation | Accepted pairs | Gradient steps |
|---|---|---|---|
| Original Condition D (collapsed) | 1 | — | **288** |
| Original Condition A (stable) | 1 | 94 | 282 |
| New FRESH_SMALL (stable) | 1 | 63 | **63** |
| New FRESH_SMALL (stable) | 2 | 69 | **69** |
| New REC_BIG / FRESH_BIG (stable) | 1 | 74 / 63 | 222 / 189 |

**The new FRESH_SMALL received roughly 4x fewer gradient steps per generation than the
original Condition D** (63-69 vs. 288). This is because the original D was explicitly
designed to *match Condition A's total step count* by repeatedly sampling fresh small
batches until hitting a target (up to 43 iterations, seeing 500+ distinct questions per
generation). This follow-up's SMALL conditions instead do a single pass through one fixed
120-question pool — a much smaller total training volume, with no mechanism to reach a
matched step-count target. **In fixing the recursion/optimization confound, this design
introduced a new, unintended difference in total training volume** — and, within the new
2×2 itself, BIG conditions also get ~3x more total gradient steps than SMALL conditions
(189-222 vs. 63-74) purely from the 3-epoch structure, an additional internal confound.

## Honest interpretation

This result does not cleanly settle whether the original Condition D's collapse was about
recursion or optimization dynamics, because a third variable (total gradient steps per
generation) moved along with the manipulation. What it does show: **at a substantially
reduced total training volume, none of the four recursion x optimization combinations
collapse.** The most defensible reading, combining this with the original Phase E result,
is that **training volume/duration is a plausible third candidate driving the original
collapse** — consistent with the RL-instability literature the publication roadmap already
flagged (repeated small-batch updates compounding instability over many steps, e.g. Tang et
al. 2405.08448, Zheng et al. 2307.04964's PPO-collapse discussion) — rather than recursion
or batch size being the sole cause.

## What a fully clean follow-up would need

Hold **total gradient step count constant** across all four cells while independently
varying recursion (frozen vs. fresh pool) and update structure (batch size / epochs vs.
single-pass). Concretely: give SMALL conditions enough additional small-batch iterations
(sampling further chunks, or repeating the pool) to match BIG's ~200+ step count, rather
than stopping after one pass through a fixed-size pool. This is the natural next
refinement, not yet run.

## Status

This is now the second experiment (after the original Phase E) to probe this question, and
the second to leave it not fully resolved — but each attempt has narrowed the space of
candidate explanations further (originally: recursion vs. optimization dynamics vs. data
source, now: those plus training volume). Recorded here rather than treated as a dead end,
since the narrowing itself is a real, honest research contribution, and the exact
step-count-matched design needed next is now precisely specified.

**Code:** `src/phase_e_2x2.py`, `src/phase_e_2x2_evaluate.py`. **Results:**
`results/phase_e_2x2_evaluations.pkl`.
