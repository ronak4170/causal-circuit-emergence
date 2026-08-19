"""
Pythia-410M version of phase_b_setup.py: load the final Phase A Pythia
checkpoint (iteration 135, matching the GPT-2 convention -- final checkpoint,
no single unambiguous transition point identified) and verify it performs
above chance before circuit-hunting.
"""
import pickle
import random

import torch

import pythia_compat  # noqa: F401 -- side-effecting import, must come before HookedTransformer
from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance
from rft_training_loop import format_prompt, extract_predicted_probability, is_correct

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "EleutherAI/pythia-410m"
POST_TRANSITION_STEP = 135
CHECKPOINT_PATH = f"results/phase_a_pythia_checkpoints/ckpt_{POST_TRANSITION_STEP:04d}.pt"


def load_model():
    model = HookedTransformer.from_pretrained(MODEL_NAME).to(DEVICE)
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
    print(f"Config: n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}, "
          f"d_model={model.cfg.d_model}")

    log = pickle.load(open("results/phase_a_pythia_log.pkl", "rb"))
    idx = log["iteration"].index(POST_TRANSITION_STEP)
    print(f"Logged intervention accuracy at this step (single iteration, n=4): "
          f"{log['interv_acc'][idx]:.3f}")

    acc = eval_intervention_accuracy(model, n=20)
    print(f"Fresh eval intervention accuracy (n=20, tolerance=0.10): {acc:.3f}")
    chance_note = ("well above the ~0 rate a random/unparseable-percentage model would "
                    "get" if acc > 0.30 else "NOT convincingly above chance -- STOP, "
                    "reconsider checkpoint choice before proceeding to circuit-hunting")
    print(chance_note)
