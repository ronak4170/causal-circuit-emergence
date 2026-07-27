"""
Step 1: load the Phase A checkpoint selected in results/phase_a_conclusion.md
(iteration 135, the final checkpoint -- no single unambiguous "post-transition"
point existed, see the conclusion doc for why) and verify it actually performs
above chance on intervention questions before spending time on circuit-hunting.

Adaptation note: Phase A's checkpoints are saved as individual half-precision
files in results/phase_a_checkpoints/ckpt_XXXX.pt (see phase_a_main.py's
disk-space fix), not as one results/phase_a_checkpoints.pkl dict as the
starter code assumed -- loading is adjusted accordingly.
"""
import pickle
import random

import torch
from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance
from rft_training_loop import format_prompt, extract_predicted_probability, is_correct

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MPS excluded from auto-detection: TransformerLens warns it may give silently
# incorrect results (see docs/phase0_setup.md).

POST_TRANSITION_STEP = 135  # from results/phase_a_conclusion.md
CHECKPOINT_PATH = f"results/phase_a_checkpoints/ckpt_{POST_TRANSITION_STEP:04d}.pt"


def load_model():
    model = HookedTransformer.from_pretrained("gpt2").to(DEVICE)
    half_state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    full_state = {k: v.float() for k, v in half_state.items()}
    model.load_state_dict(full_state)
    model.eval()
    return model


def eval_intervention_accuracy(model, n=20, seed=999):
    rng = random.Random(seed)
    topologies = ["chain", "fork", "collider", "confounded"]
    n_correct = 0
    for _ in range(n):
        topo = rng.choice(topologies)
        inst = generate_instance(topo, "intervention", rng)
        prompt = format_prompt(inst["question"])
        tokens = model.to_tokens(prompt)
        generated = model.generate(tokens, max_new_tokens=15, temperature=0.8,
                                    do_sample=True, verbose=False)
        new_tokens = generated[0, tokens.shape[1]:]
        gen_text = model.to_string(new_tokens)
        pred = extract_predicted_probability(gen_text)
        if is_correct(pred, inst["oracle_answer"], tolerance=0.10):
            n_correct += 1
    return n_correct / n


if __name__ == "__main__":
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    model = load_model()
    print(f"Loaded post-transition checkpoint at step {POST_TRANSITION_STEP}")

    log = pickle.load(open("results/phase_a_log.pkl", "rb"))
    idx = log["iteration"].index(POST_TRANSITION_STEP)
    print(f"Logged intervention accuracy at this step (single iteration, n=4): "
          f"{log['interv_acc'][idx]:.3f}")

    acc = eval_intervention_accuracy(model, n=20)
    print(f"Fresh eval intervention accuracy (n=20, tolerance=0.10): {acc:.3f}")
    chance_note = ("well above the ~0 rate a random/unparseable-percentage model would "
                    "get" if acc > 0.30 else "NOT convincingly above chance -- STOP, "
                    "reconsider checkpoint choice before proceeding to circuit-hunting")
    print(chance_note)
