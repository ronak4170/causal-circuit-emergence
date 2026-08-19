"""
Phase F: RFT training starting from a Phase F warm-up checkpoint (rationale
or control condition -- see warmup_sft_rationale.py / warmup_sft_intervention_
control.py). Identical curriculum/hyperparameters to the original Phase A
(150 iterations, 4 questions/rung/iteration) for comparability, except
max_new_tokens=60 (vs. the original 15) to leave room for a rationale before
the final percentage answer -- this also applies uniformly to the control
condition's rollouts, even though its warm-up never produces rationale text,
so the only Phase-F-vs-original-Phase-A confound is "which warm-up checkpoint
started training," selected via CONDITION below.

Run with: python phase_f_main.py rationale   OR   python phase_f_main.py control
"""
import os
import pickle
import random
import sys

import torch

from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance
from rft_training_loop import rft_iteration, finetune_on_accepted

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"

TOPOLOGIES = ["chain", "fork", "collider", "confounded"]
RUNGS = ["association", "intervention", "counterfactual"]

N_ITERATIONS = 150
N_PER_RUNG = 4
CHECKPOINT_EVERY = 15
N_SAMPLES_PER_QUESTION = 4
TEMPERATURE = 0.8
LR = 1e-5
MAX_NEW_TOKENS = 60


def build_task_batch(n_per_rung, seed):
    rng = random.Random(seed)
    batch = []
    for rung in RUNGS:
        for _ in range(n_per_rung):
            topo = rng.choice(TOPOLOGIES)
            inst = generate_instance(topo, rung, rng)
            batch.append({"question": inst["question"], "oracle_answer": inst["oracle_answer"],
                          "rung": rung, "topology": topo})
    return batch


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("rationale", "control"):
        print("Usage: python phase_f_main.py [rationale|control]")
        sys.exit(1)
    condition = sys.argv[1]

    warmup_path = f"results/warmup_model_{'rationale' if condition == 'rationale' else 'intervention_control'}.pt"
    checkpoint_dir = f"results/phase_f_{condition}_checkpoints"
    log_path = f"results/phase_f_{condition}_log.pkl"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Condition: {condition}. Loading warm-started {MODEL_NAME} from {warmup_path}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(warmup_path, map_location=DEVICE))
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    log = {"iteration": [], "assoc_acc": [], "interv_acc": [], "cf_acc": [],
           "n_accepted": [], "finetune_loss": []}

    for iteration in range(N_ITERATIONS):
        task_batch = build_task_batch(N_PER_RUNG, seed=iteration)
        accepted_pairs, stats = rft_iteration(
            model, task_batch, DEVICE, tolerance=0.10,
            n_samples_per_question=N_SAMPLES_PER_QUESTION, temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS)
        loss = finetune_on_accepted(model, optimizer, accepted_pairs, DEVICE)

        assoc_acc = stats["association"][0] / max(stats["association"][1], 1)
        interv_acc = stats["intervention"][0] / max(stats["intervention"][1], 1)
        cf_acc = stats["counterfactual"][0] / max(stats["counterfactual"][1], 1)

        print(f"Iter {iteration}: assoc={assoc_acc:.2f} interv={interv_acc:.2f} "
              f"cf={cf_acc:.2f} n_accepted={len(accepted_pairs)} loss={loss}")

        log["iteration"].append(iteration)
        log["assoc_acc"].append(assoc_acc)
        log["interv_acc"].append(interv_acc)
        log["cf_acc"].append(cf_acc)
        log["n_accepted"].append(len(accepted_pairs))
        log["finetune_loss"].append(loss)

        if iteration % CHECKPOINT_EVERY == 0:
            half_state = {k: v.detach().cpu().half().clone() for k, v in model.state_dict().items()}
            torch.save(half_state, f"{checkpoint_dir}/ckpt_{iteration:04d}.pt")

        if iteration % 10 == 0:
            pickle.dump(log, open(log_path, "wb"))

    pickle.dump(log, open(log_path, "wb"))
    print(f"Done. Saved {log_path} and checkpoints to {checkpoint_dir}/")


if __name__ == "__main__":
    main()
