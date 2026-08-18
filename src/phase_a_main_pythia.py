"""
Pythia-410M version of phase_a_main.py -- RFT training to produce a
competent checkpoint for the Phase B/C/D scale-up replication. Same
curriculum/hyperparameters as the original GPT-2 Phase A run (150
iterations, 4 questions/rung/iteration) for maximum comparability; only the
model and checkpoint paths differ.
"""
import os
import pickle
import random

import torch

import pythia_compat  # noqa: F401 -- side-effecting import, must come before HookedTransformer
from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance
from rft_training_loop import rft_iteration, finetune_on_accepted

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "EleutherAI/pythia-410m"

TOPOLOGIES = ["chain", "fork", "collider", "confounded"]
RUNGS = ["association", "intervention", "counterfactual"]

N_ITERATIONS = 150
N_PER_RUNG = 4
CHECKPOINT_EVERY = 15
CHECKPOINT_DIR = "results/phase_a_pythia_checkpoints"
N_SAMPLES_PER_QUESTION = 4
TEMPERATURE = 0.8
LR = 1e-5


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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Loading warm-started {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load("results/warmup_model_pythia.pt", map_location=DEVICE))
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    log = {"iteration": [], "assoc_acc": [], "interv_acc": [], "cf_acc": [],
           "n_accepted": [], "finetune_loss": []}

    for iteration in range(N_ITERATIONS):
        task_batch = build_task_batch(N_PER_RUNG, seed=iteration)
        accepted_pairs, stats = rft_iteration(
            model, task_batch, DEVICE, tolerance=0.10,
            n_samples_per_question=N_SAMPLES_PER_QUESTION, temperature=TEMPERATURE)
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
            torch.save(half_state, f"{CHECKPOINT_DIR}/ckpt_{iteration:04d}.pt")

        if iteration % 10 == 0:
            pickle.dump(log, open("results/phase_a_pythia_log.pkl", "wb"))

    pickle.dump(log, open("results/phase_a_pythia_log.pkl", "wb"))
    print(f"Done. Saved results/phase_a_pythia_log.pkl and checkpoints to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
