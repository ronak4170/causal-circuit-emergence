"""
Step 4: for each Phase A checkpoint x prompt pair x prompt version, evaluate
and compute the implicit/explicit accuracy gap per topology.

Checkpoint coverage: all 10 available Phase A checkpoints (0, 15, 30, ...,
135). Phase A found no single clean transition (see phase_a_conclusion.md),
so per Step 3.2's alternative guidance ("if Phase A found no clean phase
transition, spread checkpoints roughly evenly across training instead"),
every saved checkpoint is evaluated rather than concentrating density around
one hypothesized moment.

Design decisions, made explicitly per Step 4.2:
- Best-of across explicit variants: a pair counts as "explicit-correct" if
  ANY of the 3 explicit variants is correct -- the strongest fair test of
  whether scaffolding CAN rescue the model.
- Tolerance: 0.10, matching Phase A/B/C throughout.
- Evaluation method: single-forward-pass greedy decoding (reusing
  phase_c_ablation.predicted_percentage_greedy) rather than multi-token
  .generate(), for the same reliability/cost reasons disclosed in Phase C.
"""
import os
import pickle

import torch
from transformer_lens import HookedTransformer

from phase_c_ablation import predicted_percentage_greedy
from phase_d_prompt_pairs import build_prompt_pair_set
from rft_training_loop import format_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "results/phase_a_checkpoints"
TOLERANCE = 0.10


def is_correct(pred, true, tolerance=TOLERANCE):
    return pred is not None and abs(pred - true) <= tolerance


def evaluate_checkpoint(model, prompt_pairs):
    implicit_correct = {}
    explicit_correct = {}
    for pair in prompt_pairs:
        topo = pair["topology"]
        implicit_correct.setdefault(topo, [])
        explicit_correct.setdefault(topo, [])

        imp_prompt = format_prompt(pair["implicit"]) + " Answer:"
        imp_pred = predicted_percentage_greedy(model, imp_prompt)
        implicit_correct[topo].append(is_correct(imp_pred, pair["true_answer"]))

        exp_hits = []
        for exp_text in pair["explicit_variants"]:
            exp_prompt = format_prompt(exp_text) + " Answer:"
            exp_pred = predicted_percentage_greedy(model, exp_prompt)
            exp_hits.append(is_correct(exp_pred, pair["true_answer"]))
        explicit_correct[topo].append(any(exp_hits))

    summary = {"implicit": {}, "explicit": {}, "gap": {}, "n": {}}
    for topo in implicit_correct:
        i_acc = sum(implicit_correct[topo]) / len(implicit_correct[topo])
        e_acc = sum(explicit_correct[topo]) / len(explicit_correct[topo])
        summary["implicit"][topo] = i_acc
        summary["explicit"][topo] = e_acc
        summary["gap"][topo] = e_acc - i_acc
        summary["n"][topo] = len(implicit_correct[topo])
    summary["raw"] = {"implicit": implicit_correct, "explicit": explicit_correct}
    return summary


def main():
    print("Building prompt pair set...")
    prompt_pairs = build_prompt_pair_set(n_per_topology=15, seed=1234)
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
        print(f"Step {step}: implicit={summary['implicit']}, explicit={summary['explicit']}, "
              f"gap={summary['gap']}")

    pickle.dump(per_checkpoint, open("results/phase_d_results.pkl", "wb"))
    print("\nSaved results/phase_d_results.pkl")


if __name__ == "__main__":
    main()
