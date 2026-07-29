"""
Follow-up to Phase D: re-run the evaluation with a third condition
(neutral_control, length-matched to explicit but content-free) across all
10 checkpoints, to separate length from scaffolding-content effects.
"""
import os
import pickle

import torch
from transformer_lens import HookedTransformer

from phase_c_ablation import predicted_percentage_greedy
from phase_d_length_control import build_prompt_pair_set_with_control
from rft_training_loop import format_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "results/phase_a_checkpoints"
TOLERANCE = 0.10


def is_correct(pred, true, tolerance=TOLERANCE):
    return pred is not None and abs(pred - true) <= tolerance


def evaluate_checkpoint(model, prompt_pairs):
    correct = {"implicit": {}, "explicit": {}, "neutral_control": {}}
    for pair in prompt_pairs:
        topo = pair["topology"]
        for cond in correct:
            correct[cond].setdefault(topo, [])

        imp_prompt = format_prompt(pair["implicit"]) + " Answer:"
        imp_pred = predicted_percentage_greedy(model, imp_prompt)
        correct["implicit"][topo].append(is_correct(imp_pred, pair["true_answer"]))

        neutral_prompt = format_prompt(pair["neutral_control"]) + " Answer:"
        neutral_pred = predicted_percentage_greedy(model, neutral_prompt)
        correct["neutral_control"][topo].append(is_correct(neutral_pred, pair["true_answer"]))

        exp_hits = []
        for exp_text in pair["explicit_variants"]:
            exp_prompt = format_prompt(exp_text) + " Answer:"
            exp_pred = predicted_percentage_greedy(model, exp_prompt)
            exp_hits.append(is_correct(exp_pred, pair["true_answer"]))
        correct["explicit"][topo].append(any(exp_hits))

    summary = {cond: {} for cond in correct}
    for cond in correct:
        for topo in correct[cond]:
            summary[cond][topo] = sum(correct[cond][topo]) / len(correct[cond][topo])
    return summary


def main():
    print("Building prompt pair set with length-matched neutral control...")
    prompt_pairs = build_prompt_pair_set_with_control(n_per_topology=15, seed=1234)
    print(f"Total pairs: {len(prompt_pairs)}")

    ckpt_files = sorted(f for f in os.listdir(CHECKPOINT_DIR)
                         if f.startswith("ckpt_") and f.endswith(".pt"))
    steps = [int(f[len("ckpt_"):-len(".pt")]) for f in ckpt_files]
    print(f"Evaluating checkpoints: {steps}")

    print("Loading model shell...")
    model = HookedTransformer.from_pretrained("gpt2").to(DEVICE)

    per_checkpoint = {}
    for step, fname in zip(steps, ckpt_files):
        half_state = torch.load(f"{CHECKPOINT_DIR}/{fname}", map_location=DEVICE)
        full_state = {k: v.float() for k, v in half_state.items()}
        model.load_state_dict(full_state)
        model.eval()

        summary = evaluate_checkpoint(model, prompt_pairs)
        per_checkpoint[step] = summary
        print(f"Step {step}: implicit={summary['implicit']}, "
              f"neutral_control={summary['neutral_control']}, explicit={summary['explicit']}")

    pickle.dump(per_checkpoint, open("results/phase_d_control_results.pkl", "wb"))
    print("\nSaved results/phase_d_control_results.pkl")


if __name__ == "__main__":
    main()
