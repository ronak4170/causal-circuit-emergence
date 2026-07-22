"""
Main Phase A experiment script: train with RFT across a curriculum of all
three rungs, checkpoint regularly, and log per-rung accuracy.
LLC estimation is done as a SEPARATE post-hoc pass over saved checkpoints
(see llc_estimation_phase_a.py) rather than inline, to keep the training
loop fast.

Scale note: N_ITERATIONS and batch sizes are calibrated for a single CPU
(no cluster GPU available for this run -- see docs/phase0_setup.md), per the
starter code's own guidance to "tune based on wall-clock budget." This is a
smaller-scale run than a GPU budget would allow; that constraint is reported
explicitly in results/phase_a_conclusion.md rather than silently assumed away.
"""
import os
import pickle
import random

import torch
from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance
from rft_training_loop import rft_iteration, finetune_on_accepted

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MPS is intentionally excluded from auto-detection: TransformerLens warns it "may
# produce silently incorrect results" (see docs/phase0_setup.md). CUDA has no such
# documented issue, so it's used automatically when available (e.g. on Colab).

TOPOLOGIES = ["chain", "fork", "collider", "confounded"]
RUNGS = ["association", "intervention", "counterfactual"]

N_ITERATIONS = 150
N_PER_RUNG = 4  # 4 questions x 3 rungs = 12 questions per iteration
# Checkpointing note: an earlier version of this script accumulated full-precision
# (fp32, ~630MB each) GPT-2 state dicts in one in-memory dict and re-pickled the
# WHOLE growing dict to disk every 10 iterations -- this filled the disk (12GB+)
# and crashed the run at iteration 60. Fixed by writing each checkpoint as its own
# half-precision file, written once, immediately, to a dedicated directory.
CHECKPOINT_EVERY = 15  # -> 10 checkpoints total over 150 iterations
CHECKPOINT_DIR = "results/phase_a_checkpoints"
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

    print("Loading warm-started model...")
    model = HookedTransformer.from_pretrained("gpt2").to(DEVICE)
    model.load_state_dict(torch.load("results/warmup_model.pt", map_location=DEVICE))
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
            pickle.dump(log, open("results/phase_a_log.pkl", "wb"))

    pickle.dump(log, open("results/phase_a_log.pkl", "wb"))
    print(f"Done. Saved results/phase_a_log.pkl and checkpoints to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
