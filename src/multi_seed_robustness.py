"""
Multi-seed robustness check for the paper's two headline quantitative
claims, addressing the "single-seed, small-n" weakness flagged in the
publication roadmap:
1. Phase B's cross-topology transfer (90-98% restoration) -- rerun with 5
   different random seeds for the held-out test-pair generation (same
   candidate circuit, same checkpoint), report mean +/- std across seeds.
2. Phase D's discrimination-test correlation (r=0.985-0.998 on
   chain/fork/collider) -- same idea, 5 seeds.
No retraining involved -- both are evaluation-only against the existing
iteration-135 checkpoint, so this runs entirely on CPU with no GPU/Colab risk.
"""
import pickle

import numpy as np

from phase_b_setup import load_model
from phase_b_cross_topology import CANDIDATE_HEADS, CANDIDATE_MLP_LAYERS, score_pair_with_circuit
from phase_b_prompt_pairs import build_pair_batch, TOPOLOGY_VAR_MAP
from phase_d_discrimination_test import build_discrimination_test_set, pearson_corr
from phase_c_ablation import predicted_percentage_greedy
from rft_training_loop import format_prompt

SEEDS = [42, 101, 202, 303, 404]


def run_phase_b_multiseed(model):
    print("\n=== Phase B cross-topology transfer, multi-seed ===")
    per_seed = {topo: [] for topo in TOPOLOGY_VAR_MAP}
    for seed in SEEDS:
        pairs, _ = build_pair_batch(n_per_topology=15, seed=seed)
        for topo in TOPOLOGY_VAR_MAP:
            topo_pairs = [p for p in pairs if p["topology"] == topo]
            scores = []
            for pair in topo_pairs:
                result = score_pair_with_circuit(model, pair, CANDIDATE_HEADS, CANDIDATE_MLP_LAYERS)
                if result is not None:
                    scores.append(result[1])
            mean_score = sum(scores) / len(scores) if scores else None
            per_seed[topo].append(mean_score)
        print(f"  seed={seed}: " + ", ".join(f"{t}={per_seed[t][-1]:.3f}" for t in TOPOLOGY_VAR_MAP))

    summary = {}
    for topo, vals in per_seed.items():
        vals = [v for v in vals if v is not None]
        summary[topo] = {"mean": np.mean(vals), "std": np.std(vals), "values": vals}
    print("\nSummary (mean +/- std across 5 seeds):")
    for topo, s in summary.items():
        print(f"  {topo}: {s['mean']:.3f} +/- {s['std']:.3f}  (values: {[round(v,3) for v in s['values']]})")
    return summary


def run_phase_d_multiseed(model):
    print("\n=== Phase D discrimination correlation, multi-seed ===")
    per_seed = {"chain": [], "fork": [], "collider": []}
    for seed in SEEDS:
        test_set = build_discrimination_test_set(n_per_topology=25, seed=seed)
        from collections import defaultdict
        by_topo = defaultdict(list)
        for q in test_set:
            prompt = format_prompt(q["prompt"]) + " Answer:"
            pred = predicted_percentage_greedy(model, prompt)
            if pred is not None:
                by_topo[q["topology"]].append((pred, q["true_answer"]))
        for topo in per_seed:
            pairs = by_topo.get(topo, [])
            if len(pairs) >= 2:
                preds, trues = zip(*pairs)
                corr = pearson_corr(list(preds), list(trues))
            else:
                corr = None
            per_seed[topo].append(corr)
        print(f"  seed={seed}: " + ", ".join(f"{t}={per_seed[t][-1]}" for t in per_seed))

    summary = {}
    for topo, vals in per_seed.items():
        vals = [v for v in vals if v is not None]
        summary[topo] = {"mean": np.mean(vals), "std": np.std(vals), "values": vals}
    print("\nSummary (mean +/- std across 5 seeds):")
    for topo, s in summary.items():
        print(f"  {topo}: {s['mean']:.4f} +/- {s['std']:.4f}  (values: {[round(v,4) for v in s['values']]})")
    return summary


def main():
    print("Loading checkpoint...")
    model = load_model()

    b_summary = run_phase_b_multiseed(model)
    d_summary = run_phase_d_multiseed(model)

    pickle.dump({"phase_b": b_summary, "phase_d": d_summary},
                open("results/multi_seed_robustness.pkl", "wb"))
    print("\nSaved results/multi_seed_robustness.pkl")


if __name__ == "__main__":
    main()
