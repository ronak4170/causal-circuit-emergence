"""
Phase E: recursive RL fine-tuning across 4 conditions, 3 generations each,
starting from the Phase A iteration-135 checkpoint.

Condition semantics (corrected from the starter pseudocode -- see
PREREGISTRATION.md's Phase E section for why the original pseudocode's A and
D were mechanistically identical and wouldn't have isolated recursion at all):

- A (vanilla recursive): generation g's ENTIRE training corpus is a fixed
  snapshot of generation (g-1)'s own oracle-filtered rollouts on a batch of
  questions sampled ONCE for that generation. No fresh oracle content enters
  after that batch is drawn.
- B (real-data-anchored): same as A, plus ~7.5% freshly oracle-labeled pairs
  mixed into the training corpus.
- C (diversity-filtered): same as A, but caps accepted rollouts per
  (topology, rung) bucket before fine-tuning.
- D (no-recursion control): continues Phase A's ORIGINAL loop exactly --
  fresh oracle question batches every iteration, same as phase_a_main.py --
  for a comparable number of gradient updates. The question pool never stops
  refreshing, so there's no generation-to-generation self-referential corpus.
"""
import os
import pickle
import random
from collections import defaultdict

import torch
from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance
from phase_b_setup import load_model
from rft_training_loop import rft_iteration, finetune_on_accepted, format_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "results/phase_e_checkpoints"
TOPOLOGIES = ["chain", "fork", "collider", "confounded"]
RUNGS = ["association", "intervention", "counterfactual"]

N_QUESTIONS_PER_GEN = 180  # ~15/topology/rung
N_EPOCHS = 3
N_SAMPLES_PER_QUESTION = 4
TEMPERATURE = 0.8
TOLERANCE = 0.10
LR = 1e-5
REAL_DATA_FRACTION_B = 0.075
MAX_PER_BUCKET_C = 5


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


def oracle_pair(item):
    """Build a (prompt, generation_text) pair using the oracle's own answer,
    in the same format finetune_on_accepted expects (prompt ends before the
    answer; generation_text is what a correct model completion looks like)."""
    pct = max(0, min(100, round(item["oracle_answer"] * 100 / 5) * 5))
    prompt = format_prompt(item["question"])
    return (prompt, f" Answer: {pct}%")


def run_generation_recursive(model, generation_num, condition):
    """Conditions A, B, C: fixed self-generated corpus per generation."""
    seed = generation_num * 7919
    task_batch = build_diverse_task_batch(N_QUESTIONS_PER_GEN, seed=seed)

    accepted_pairs, stats = rft_iteration(
        model, task_batch, DEVICE, tolerance=TOLERANCE,
        n_samples_per_question=N_SAMPLES_PER_QUESTION, temperature=TEMPERATURE)

    # Need to know which (topology, rung) each accepted pair came from, for
    # condition C's bucket cap. rft_iteration doesn't return this mapping, so
    # re-derive it by re-matching prompts (accepted_pairs preserves prompt
    # identity from format_prompt(item["question"])).
    prompt_to_item = {format_prompt(item["question"]): item for item in task_batch}

    if condition == "C":
        buckets = defaultdict(list)
        capped = []
        for prompt, gen in accepted_pairs:
            item = prompt_to_item.get(prompt)
            key = (item["topology"], item["rung"]) if item else ("unknown", "unknown")
            if len(buckets[key]) < MAX_PER_BUCKET_C:
                buckets[key].append((prompt, gen))
                capped.append((prompt, gen))
        accepted_pairs = capped

    if condition == "B":
        n_fresh = int(len(accepted_pairs) * REAL_DATA_FRACTION_B / (1 - REAL_DATA_FRACTION_B))
        fresh_batch = build_diverse_task_batch(max(1, n_fresh), seed=seed + 1)
        fresh_pairs = [oracle_pair(item) for item in fresh_batch]
        accepted_pairs = accepted_pairs + fresh_pairs

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = 0
    for _ in range(N_EPOCHS):
        random.Random(seed).shuffle(accepted_pairs)
        finetune_on_accepted(model, optimizer, accepted_pairs, DEVICE)
        total_steps += len(accepted_pairs)

    return model, {"stats": stats, "n_accepted": len(accepted_pairs), "n_gradient_steps": total_steps}


def run_generation_control_D(model, generation_num, n_gradient_steps_target):
    """Condition D: extend Phase A's original loop -- fresh oracle batches
    every iteration, no self-referential corpus -- for a comparable number of
    gradient updates to what the recursive conditions used this generation."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = 0
    iteration = 0
    while total_steps < n_gradient_steps_target:
        seed = generation_num * 7919 + 1000 + iteration
        task_batch = build_diverse_task_batch(12, seed=seed)  # matches Phase A's per-iter size
        accepted_pairs, stats = rft_iteration(
            model, task_batch, DEVICE, tolerance=TOLERANCE,
            n_samples_per_question=N_SAMPLES_PER_QUESTION, temperature=TEMPERATURE)
        finetune_on_accepted(model, optimizer, accepted_pairs, DEVICE)
        total_steps += len(accepted_pairs)
        iteration += 1

    return model, {"n_iterations": iteration, "n_gradient_steps": total_steps}


def save_checkpoint(model, condition, generation):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    half_state = {k: v.detach().cpu().half().clone() for k, v in model.state_dict().items()}
    torch.save(half_state, f"{CHECKPOINT_DIR}/condition_{condition}_gen_{generation}.pt")


def run_condition(condition, n_generations=3):
    print(f"\n=== Condition {condition} ===")
    model = load_model()  # starts from iteration-135 checkpoint
    save_checkpoint(model, condition, 0)

    log = []
    for gen in range(1, n_generations + 1):
        if condition == "D":
            # Match D's step count to condition A's same-generation step count,
            # read back from a previously-run A log if available, else use the
            # nominal target (N_QUESTIONS_PER_GEN accepted-rate estimate * N_EPOCHS).
            target_steps = log_lookup_target_steps(gen)
            model, info = run_generation_control_D(model, gen, target_steps)
        else:
            model, info = run_generation_recursive(model, gen, condition)
        save_checkpoint(model, condition, gen)
        log.append({"generation": gen, **info})
        print(f"Condition {condition} generation {gen} complete: {info}")
        # Save incrementally after each generation, not just at the end -- an
        # earlier Phase E run lost condition D's entire log to a session
        # interruption because this only wrote once, after the full loop
        # finished. Same lesson as Phase A's disk-space checkpoint fix.
        pickle.dump(log, open(f"results/phase_e_condition_{condition}_log.pkl", "wb"))

    return log


_TARGET_STEPS_CACHE = {}


def log_lookup_target_steps(gen):
    """Read condition A's gradient-step count for this generation, so D can
    match it. Falls back to a nominal estimate if A hasn't been run yet."""
    if gen in _TARGET_STEPS_CACHE:
        return _TARGET_STEPS_CACHE[gen]
    try:
        a_log = pickle.load(open("results/phase_e_condition_A_log.pkl", "rb"))
        entry = next(e for e in a_log if e["generation"] == gen)
        steps = entry["n_gradient_steps"]
    except (FileNotFoundError, StopIteration):
        steps = int(N_QUESTIONS_PER_GEN * 0.65 * N_EPOCHS)  # nominal estimate
    _TARGET_STEPS_CACHE[gen] = steps
    return steps


if __name__ == "__main__":
    import sys
    conditions = sys.argv[1:] if len(sys.argv) > 1 else ["A", "B", "C", "D"]
    for cond in conditions:
        run_condition(cond, n_generations=3)
