"""
Follow-up to Phase E, addressing the disclosed confound in results/phase_e_conclusion.md:
the original Condition A (recursive) and Condition D (non-recursive control) differed in
TWO ways at once -- data source AND optimization dynamics (batch size / update
frequency) -- so the original result couldn't cleanly attribute D's collapse to either
factor alone.

This implements a proper 2x2, crossing:

FACTOR 1 -- RECURSION (is the question pool frozen and reused across generations, or
fresh each generation?):
  - FROZEN: the SAME 120 questions (fixed seed, generated once) are used at every
    generation. The model regenerates its own (possibly different) answers to them each
    time, but the TASK DISTRIBUTION itself never refreshes -- this is the literature's
    actual definition of recursive self-training (Shumailov et al.): each generation
    trains on a self-referential snapshot, not a broadening task distribution.
  - FRESH: a newly-sampled set of 120 questions (new seed) is used every generation --
    the model keeps encountering genuinely novel tasks, matching continued/non-recursive
    RL post-training.

FACTOR 2 -- OPTIMIZATION DYNAMICS (how is a generation's question pool consumed?):
  - BIG_EPOCH: one large batch, accepted rollouts collected once, then 3 clean epochs of
    fine-tuning over that fixed accepted set (matches the original Phase E Conditions
    A/B/C's mechanism).
  - SMALL_ITER: the pool is split into small chunks (size 12); loop through chunks doing
    accept-and-immediately-fine-tune, single pass, no repeats (matches the original Phase
    E Condition D's mechanism).

Both factors are now independently controlled -- unlike the original Phase E design where
"recursive" (A) always meant big-batch-epoch and "non-recursive" (D) always meant
small-batch-iterative.

Scale reduced from the original Phase E run (120 questions/generation instead of 180, 2
generations instead of 3) to keep this tractable given it's now 4 NEW conditions.
"""
import os
import pickle
import random

import torch
from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance
from phase_b_setup import load_model
from rft_training_loop import rft_iteration, finetune_on_accepted

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "results/phase_e_2x2_checkpoints"
TOPOLOGIES = ["chain", "fork", "collider", "confounded"]
RUNGS = ["association", "intervention", "counterfactual"]

N_QUESTIONS_PER_GEN = 120
N_GENERATIONS = 2
N_EPOCHS = 3
SMALL_BATCH_SIZE = 12
N_SAMPLES_PER_QUESTION = 4
TEMPERATURE = 0.8
TOLERANCE = 0.10
LR = 1e-5
FROZEN_POOL_SEED = 999  # the ONE seed reused every generation for FROZEN conditions

CONDITIONS = ["REC_BIG", "REC_SMALL", "FRESH_BIG", "FRESH_SMALL"]


def build_diverse_task_batch(n_questions, seed):
    rng = random.Random(seed)
    batch = []
    for _ in range(n_questions):
        topo = rng.choice(TOPOLOGIES)
        rung = rng.choice(RUNGS)
        inst = generate_instance(topo, rung, rng)
        batch.append({"question": inst["question"], "oracle_answer": inst["oracle_answer"],
                       "rung": rung, "topology": topo})
    return batch


def get_question_pool(condition, generation_num):
    """FACTOR 1: recursion. FROZEN conditions reuse the same seed every
    generation (same 120 questions every time); FRESH conditions use a new
    seed each generation."""
    if condition.startswith("REC"):
        seed = FROZEN_POOL_SEED
    else:
        seed = generation_num * 7919
    return build_diverse_task_batch(N_QUESTIONS_PER_GEN, seed=seed)


def run_generation(model, condition, generation_num):
    """FACTOR 2: optimization dynamics."""
    pool = get_question_pool(condition, generation_num)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = 0

    if condition.endswith("BIG"):
        accepted_pairs, stats = rft_iteration(
            model, pool, DEVICE, tolerance=TOLERANCE,
            n_samples_per_question=N_SAMPLES_PER_QUESTION, temperature=TEMPERATURE)
        for _ in range(N_EPOCHS):
            random.Random(generation_num).shuffle(accepted_pairs)
            finetune_on_accepted(model, optimizer, accepted_pairs, DEVICE)
            total_steps += len(accepted_pairs)
        n_accepted = len(accepted_pairs)

    else:  # SMALL
        n_accepted = 0
        stats = {"association": [0, 0], "intervention": [0, 0], "counterfactual": [0, 0]}
        for i in range(0, len(pool), SMALL_BATCH_SIZE):
            chunk = pool[i:i + SMALL_BATCH_SIZE]
            accepted_pairs, chunk_stats = rft_iteration(
                model, chunk, DEVICE, tolerance=TOLERANCE,
                n_samples_per_question=N_SAMPLES_PER_QUESTION, temperature=TEMPERATURE)
            finetune_on_accepted(model, optimizer, accepted_pairs, DEVICE)
            n_accepted += len(accepted_pairs)
            total_steps += len(accepted_pairs)
            for rung in stats:
                stats[rung][0] += chunk_stats[rung][0]
                stats[rung][1] += chunk_stats[rung][1]

    return model, {"stats": stats, "n_accepted": n_accepted, "n_gradient_steps": total_steps}


def save_checkpoint(model, condition, generation):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    half_state = {k: v.detach().cpu().half().clone() for k, v in model.state_dict().items()}
    torch.save(half_state, f"{CHECKPOINT_DIR}/condition_{condition}_gen_{generation}.pt")


def run_condition(condition, n_generations=N_GENERATIONS):
    print(f"\n=== Condition {condition} ===")
    model = load_model()  # starts from iteration-135 checkpoint
    save_checkpoint(model, condition, 0)

    log = []
    for gen in range(1, n_generations + 1):
        model, info = run_generation(model, condition, gen)
        save_checkpoint(model, condition, gen)
        log.append({"generation": gen, **info})
        print(f"Condition {condition} generation {gen} complete: {info}")
        pickle.dump(log, open(f"results/phase_e_2x2_condition_{condition}_log.pkl", "wb"))

    return log


if __name__ == "__main__":
    import sys
    conditions = sys.argv[1:] if len(sys.argv) > 1 else CONDITIONS
    for cond in conditions:
        run_condition(cond)
