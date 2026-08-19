"""
Pythia-410M version of phase_c_run.py: run the full Phase C ablation
experiment (mean-ablation targets, interventional test set on confounded
topology, count-matched associational test set) using Pythia's candidate
circuit.
"""
import pickle

from phase_b_setup_pythia import load_model
from phase_c_ablation_pythia import (build_diverse_prompts, compute_component_means,
                                      build_ablation_hooks, evaluate)
from phase_c_reference_answers import build_phase_c_test_set, build_associational_test_set

N_MEAN_PROMPTS = 100


def main():
    print("Loading Pythia checkpoint...")
    model = load_model()

    print(f"Building {N_MEAN_PROMPTS} diverse prompts for mean-ablation targets...")
    diverse_prompts = build_diverse_prompts(n=N_MEAN_PROMPTS, seed=7)
    head_means, mlp_means = compute_component_means(model, diverse_prompts)
    ablation_hooks = build_ablation_hooks(head_means, mlp_means)
    print(f"Computed mean-ablation targets for {len(head_means)} heads, {len(mlp_means)} MLP layers.")

    print("\nBuilding interventional test set (confounded topology, n=100)...")
    interv_test_set = build_phase_c_test_set(n_per_topology=100, seed=42, min_divergence=0.15)

    print("\nEvaluating UN-ABLATED model on interventional test set...")
    interv_unablated = evaluate(model, interv_test_set, fwd_hooks=None)
    print("Evaluating ABLATED model on interventional test set...")
    interv_ablated = evaluate(model, interv_test_set, fwd_hooks=ablation_hooks)

    counts_per_topology = {}
    for q in interv_test_set:
        counts_per_topology[q["topology"]] = counts_per_topology.get(q["topology"], 0) + 1
    print(f"\nBuilding count-matched associational test set: {counts_per_topology}")
    assoc_test_set = build_associational_test_set(counts_per_topology, seed=43)

    print("Evaluating UN-ABLATED model on associational test set...")
    assoc_unablated = evaluate(model, assoc_test_set, fwd_hooks=None)
    print("Evaluating ABLATED model on associational test set...")
    assoc_ablated = evaluate(model, assoc_test_set, fwd_hooks=ablation_hooks)

    pickle.dump(
        {
            "interv_test_set": interv_test_set, "assoc_test_set": assoc_test_set,
            "interv_unablated": interv_unablated, "interv_ablated": interv_ablated,
            "assoc_unablated": assoc_unablated, "assoc_ablated": assoc_ablated,
        },
        open("results/phase_c_pythia_raw_results.pkl", "wb"),
    )
    print("\nSaved results/phase_c_pythia_raw_results.pkl")


if __name__ == "__main__":
    main()
