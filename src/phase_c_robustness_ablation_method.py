"""
Robustness follow-up to Phase C, motivated by Miller, Chughtai & Saunders,
"Transformer Circuit Faithfulness Metrics are not Robust" (arXiv:2407.08734,
COLM 2024): faithfulness/ablation results can be highly sensitive to
seemingly minor methodology choices (mean vs. resample ablation, node vs.
edge). Phase C used MEAN ablation (mean activation across 256 diverse
prompts). This script re-runs Phase C's headline test with RESAMPLE
ablation instead -- each test question gets ablated using a SINGLE randomly
chosen different prompt's activation (not an average over many), which is
the other standard ablation methodology in the literature -- to check
whether Phase C's finding (un-ablated confounded accuracy already ~0%,
ablation collapses output to a near-constant) survives the change.
"""
import pickle
import random

import torch

from phase_b_setup import load_model
from phase_c_ablation import (build_diverse_prompts, build_ablation_hooks,
                                predicted_percentage_greedy, CANDIDATE_HEADS,
                                CANDIDATE_MLP_LAYERS, DEVICE)
from phase_c_reference_answers import build_phase_c_test_set
from rft_training_loop import format_prompt

N_RESAMPLE_POOL = 100
TOLERANCE = 0.10


def compute_single_prompt_activations(model, prompt):
    """Same shape/format as compute_component_means, but sourced from ONE
    prompt's own activations (averaged only over ITS OWN sequence positions),
    not averaged across many prompts -- the resample-ablation methodology."""
    tokens = model.to_tokens(prompt).to(DEVICE)
    _, cache = model.run_with_cache(tokens)
    head_vals = {(l, h): cache[f"blocks.{l}.attn.hook_z"][:, :, h, :].mean(dim=(0, 1))
                 for (l, h) in CANDIDATE_HEADS}
    mlp_vals = {l: cache[f"blocks.{l}.hook_mlp_out"][:, :, :].mean(dim=(0, 1))
                for l in CANDIDATE_MLP_LAYERS}
    return head_vals, mlp_vals


def evaluate_resample_ablation(model, test_questions, resample_pool, seed=42):
    """For each question, ablate using a freshly-chosen single random
    resample-source prompt (a different one per question, per standard
    resample-ablation practice)."""
    rng = random.Random(seed)
    results = []
    for q in test_questions:
        source_prompt = rng.choice(resample_pool)
        head_vals, mlp_vals = compute_single_prompt_activations(model, source_prompt)
        hooks = build_ablation_hooks(head_vals, mlp_vals)

        prompt = format_prompt(q["prompt"]) + " Answer:"
        pred = predicted_percentage_greedy(model, prompt, fwd_hooks=hooks)
        results.append({
            "topology": q["topology"], "predicted": pred,
            "interventional_true": q["interventional_answer"],
            "associational_true": q["associational_answer"],
        })
    return results


def is_correct(pred, true, tolerance=TOLERANCE):
    return pred is not None and abs(pred - true) <= tolerance


def main():
    print("Loading checkpoint...")
    model = load_model()

    print("Building Phase C interventional test set (confounded topology, n=100)...")
    test_set = build_phase_c_test_set(n_per_topology=100, seed=42, min_divergence=0.15)

    print(f"Building resample pool ({N_RESAMPLE_POOL} diverse prompts)...")
    resample_pool = build_diverse_prompts(n=N_RESAMPLE_POOL, seed=7)

    print("Evaluating under RESAMPLE ablation...")
    resample_results = evaluate_resample_ablation(model, test_set, resample_pool, seed=42)

    from collections import Counter
    pred_counts = Counter(r["predicted"] for r in resample_results)
    n_correct = sum(1 for r in resample_results if is_correct(r["predicted"], r["interventional_true"]))
    print(f"\nResample-ablation interventional accuracy: {n_correct}/{len(resample_results)} "
          f"= {n_correct/len(resample_results):.3f}")
    print(f"Distinct predicted values under resample ablation: {dict(pred_counts)}")

    # Classification: closer to associational vs interventional (same as Phase C's P1 test)
    classified = {"closer_to_associational": 0, "closer_to_interventional": 0, "tie_or_unparseable": 0}
    for r in resample_results:
        pred = r["predicted"]
        if pred is None:
            classified["tie_or_unparseable"] += 1
            continue
        d_interv = abs(pred - r["interventional_true"])
        d_assoc = abs(pred - r["associational_true"])
        if abs(d_interv - d_assoc) < 0.02:
            classified["tie_or_unparseable"] += 1
        elif d_assoc < d_interv:
            classified["closer_to_associational"] += 1
        else:
            classified["closer_to_interventional"] += 1
    print(f"Directional classification under resample ablation: {classified}")

    pickle.dump(
        {"resample_results": resample_results, "pred_counts": dict(pred_counts),
         "accuracy": n_correct / len(resample_results), "classified": classified},
        open("results/phase_c_resample_ablation_results.pkl", "wb"),
    )
    print("\nSaved results/phase_c_resample_ablation_results.pkl")


if __name__ == "__main__":
    main()
