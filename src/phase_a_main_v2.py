"""
RQ1 fix, part 2: Phase A rerun with (a) the weakened warm-up from
warmup_sft_v2.py (association starts at ~27% accuracy, not ~100%, per that
script's own checkpoint) and (b) a doubled per-iteration batch size (8
questions/rung instead of 4, i.e. 24/iteration instead of 12) to reduce the
accuracy-quantization noise that dominated the original Phase A's raw curves
(quantized to steps of 0.25 there; steps of 0.125 here).

Denser checkpointing than the original (every 10 iterations instead of 15,
so 16 checkpoints over 150 iterations instead of 10) since where any
transition might now occur is genuinely unknown -- association could show a
real emergence this time, unlike the original run where it started saturated.

Saves to SEPARATE paths from the original Phase A run (results/phase_a2_*)
so the original, already-analyzed run is not overwritten and remains
directly comparable.
"""
import os
import pickle
import random

import torch
from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance
from rft_training_loop import rft_iteration, finetune_on_accepted

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOPOLOGIES = ["chain", "fork", "collider", "confounded"]
RUNGS = ["association", "intervention", "counterfactual"]

N_ITERATIONS = 150
N_PER_RUNG = 8  # 8 questions x 3 rungs = 24 questions per iteration (was 4/12)
CHECKPOINT_EVERY = 10  # -> 16 checkpoints over 150 iterations (was 15 -> 10)
CHECKPOINT_DIR = "results/phase_a2_checkpoints"
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

    print("Loading weakly-warm-started model (v2)...")
    model = HookedTransformer.from_pretrained("gpt2").to(DEVICE)
    model.load_state_dict(torch.load("results/warmup_model_v2.pt", map_location=DEVICE))
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

        print(f"Iter {iteration}: assoc={assoc_acc:.3f} interv={interv_acc:.3f} "
              f"cf={cf_acc:.3f} n_accepted={len(accepted_pairs)} loss={loss}")

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
            pickle.dump(log, open("results/phase_a2_log.pkl", "wb"))

    pickle.dump(log, open("results/phase_a2_log.pkl", "wb"))
    print(f"Done. Saved results/phase_a2_log.pkl and checkpoints to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
