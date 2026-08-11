"""
Evaluation for the phase_e_2x2 follow-up. Same 4 metrics as phase_e_evaluate.py
(discrimination correlation, confounded floor, output diversity, ablation
qualitative check), pointed at the 2x2 checkpoint directory and condition names.
"""
import os
import pickle

import torch
from transformer_lens import HookedTransformer

from phase_e_evaluate import (discrimination_correlation, confounded_floor_check,
                                output_diversity, ablation_qualitative_check)
from phase_d_discrimination_test import build_discrimination_test_set
from phase_e_2x2 import CONDITIONS, N_GENERATIONS, CHECKPOINT_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint_model(condition, generation):
    model = HookedTransformer.from_pretrained("gpt2").to(DEVICE)
    half_state = torch.load(f"{CHECKPOINT_DIR}/condition_{condition}_gen_{generation}.pt",
                             map_location=DEVICE)
    full_state = {k: v.float() for k, v in half_state.items()}
    model.load_state_dict(full_state)
    model.eval()
    return model


def main():
    print("Building shared test sets...")
    discrimination_test_set = build_discrimination_test_set(n_per_topology=20, seed=99)

    results = {}
    for condition in CONDITIONS:
        results[condition] = {}
        for gen in range(0, N_GENERATIONS + 1):
            path = f"{CHECKPOINT_DIR}/condition_{condition}_gen_{gen}.pt"
            if not os.path.exists(path):
                print(f"Skipping condition {condition} gen {gen}: checkpoint not found")
                continue
            model = load_checkpoint_model(condition, gen)

            corrs = discrimination_correlation(model, discrimination_test_set)
            floor = confounded_floor_check(model)
            diversity = output_diversity(model)
            ablation_check = ablation_qualitative_check(model)

            results[condition][gen] = {
                "discrimination_corr": corrs, "confounded_accuracy": floor,
                "output_diversity_std": diversity, "ablation_check": ablation_check,
            }
            print(f"Condition {condition} gen {gen}: corr={corrs}, "
                  f"confounded_acc={floor:.2f}, diversity_std={diversity}, "
                  f"ablation_distinct={ablation_check['n_distinct']}")

    pickle.dump(results, open("results/phase_e_2x2_evaluations.pkl", "wb"))
    print("\nSaved results/phase_e_2x2_evaluations.pkl")


if __name__ == "__main__":
    main()
