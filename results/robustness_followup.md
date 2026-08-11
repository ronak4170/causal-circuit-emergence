# Robustness Follow-ups (Post-Phase-E)

Motivated by the publication roadmap's identified weaknesses (`docs/publication_roadmap.md`,
prioritized items 3 and 4: ablation-methodology robustness, multi-seed + CIs on headline
numbers). Both checks run entirely on CPU against the existing iteration-135 checkpoint —
no retraining, no GPU/Colab risk.

## 1. Ablation-methodology robustness (mean vs. resample ablation)

**Motivation:** Miller, Chughtai & Saunders (arXiv:2407.08734, COLM 2024) show circuit
faithfulness scores can be highly sensitive to seemingly minor ablation-methodology
choices. Phase C used mean ablation (activation averaged across 100 diverse prompts).
This reruns Phase C's headline test with **resample ablation** instead — each of the 100
confounded-topology test questions ablated using a single, freshly and independently
sampled different prompt's activation (not an average over many).

**Code:** `src/phase_c_robustness_ablation_method.py`. **Results:**
`results/phase_c_resample_ablation_results.pkl`.

**Finding: the core claim survives, but one secondary claim doesn't — and that's itself
informative.**

| | Mean ablation (Phase C) | Resample ablation (this check) |
|---|---|---|
| Interventional accuracy | 0/100 (0.0%) | 0/100 (0.0%) |
| Output pattern | Single constant (100% one value) | Mostly one value (88%), with a second value 12% of the time |
| Directional split (closer-to-associational) | Exactly 50/50 (p=0.540, not significant) | 62/100 (p=0.0105, **significant**) |

- **The core finding — ablation destroys interventional accuracy entirely — is robust
  across both ablation methodologies.** 0% under both. This is the claim that matters most
  for the paper's headline (consistency ≠ correctness), and it survives the switch to a
  different, standard ablation method.
- **The specific directional-bias question (Phase C's Prediction 1) is methodology-
  dependent.** Mean ablation showed no directional bias at all (an exact coin flip).
  Resample ablation shows a small but statistically significant associational skew
  (62%, p=0.01) — closer to what the original theory predicted, but still below the
  pre-registered 0.65 effect-size bar for "genuine support." This is a direct, concrete
  instance of exactly the phenomenon Miller/Chughtai document: the *specific* faithfulness
  conclusion (was there a directional bias?) shifted with the ablation method, even though
  the *headline* conclusion (does ablation destroy competence?) didn't.
- **Practical implication for the paper:** report both ablation methods for the directional
  claim, framed honestly as "sensitive to ablation methodology, consistent with
  Miller/Chughtai" rather than picking whichever number looks better. The 0%-accuracy floor
  claim can be stated without this caveat, since it's the one result that didn't move.

## 2. Multi-seed confidence intervals on the two headline numbers

**Motivation:** both of the paper's strongest quantitative claims were originally reported
from a single seed. Reran each with 5 independent seeds (fresh held-out test questions each
time, same checkpoint, same circuit) to get real variance estimates.

**Code:** `src/multi_seed_robustness.py`. **Results:** `results/multi_seed_robustness.pkl`.

### Phase B: cross-topology transfer (5 seeds, n=15/topology each)

| Topology | Mean ± std (5 seeds) | Original single-seed value |
|---|---|---|
| chain | 0.972 ± 0.017 | 0.981 |
| fork | 0.971 ± 0.009 | 0.974 |
| collider | 0.908 ± 0.006 | 0.904 |
| confounded | 0.976 ± 0.008 | 0.974 |

All standard deviations are small (≤0.017) relative to the effect size (restoration scores
all >0.90 vs. a 0.0 unpatched baseline and ~0.27 random-circuit baseline, per Phase B's
original random-circuit check). **The 90-98% transfer claim is not a single-seed artifact.**

### Phase D: discrimination correlation (5 seeds, n=25/topology each)

| Topology | Mean ± std (5 seeds) | Original single-seed value |
|---|---|---|
| chain | 0.9811 ± 0.0051 | 0.985 |
| fork | 0.9813 ± 0.0018 | 0.989 |
| collider | 0.9990 ± 0.0002 | 0.998 |

Even tighter than Phase B's numbers — collider's correlation varies by less than 0.001
across 5 independent seeds. **The genuine-discrimination claim (r≈0.98-0.999) is highly
stable, not a lucky draw.**

## What this changes for the publication roadmap

- Roadmap item 4 ("multi-seed + CIs on all headline numbers, ≥5 seeds") is now done for the
  two most important numbers (Phase B transfer, Phase D correlation) at the existing
  GPT-2-small scale. The roadmap's suggested seed count (5+) was met.
- Roadmap item 3 ("ablation-methodology robustness... near-zero cost, very high payoff") is
  done. The core claim held up; the directional sub-claim's methodology-sensitivity is now
  an honest, citable, expected-per-Miller/Chughtai result rather than an untested
  vulnerability.
- **Remaining from the roadmap's no-new-infrastructure items:** multi-seed CIs on Phase A's
  LLC/accuracy numbers and Phase E's discrimination/collapse numbers were not run here
  (Phase A would require full retraining per seed — not free; Phase E's numbers already
  come from comparing 4 distinct conditions, which is a different kind of robustness
  question than reseeding a fixed evaluation).
- Everything that required new GPU/Colab compute (Pythia scale-up, RQ5 2×2, RQ1 redesign)
  is still exactly as described in `docs/publication_roadmap.md` — not attempted here, per
  the earlier agreement to separate free local work from deliberate compute-budget
  decisions.
