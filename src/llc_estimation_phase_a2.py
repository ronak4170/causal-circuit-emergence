"""
LLC estimation over Phase A v2 checkpoints (the RQ1-fix rerun with a
weakened warm-up and doubled batch size -- see phase_a_main_v2.py). Same
devinterp API and settings as llc_estimation_phase_a.py, just pointed at the
new checkpoint directory (16 checkpoints instead of 10, since v2 checkpoints
every 10 iterations instead of 15).
"""
import os
import pickle
import random

import torch
from datasets import Dataset
from devinterp.slt.llc import llc
from devinterp.utils import default_nbeta
from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance
from rft_training_loop import format_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "results/phase_a2_checkpoints"
BATCH_SIZE = 16
N_LLC_EXAMPLES = 256
TOPOLOGIES = ["chain", "fork", "collider", "confounded"]
RUNGS = ["association", "intervention", "counterfactual"]


def build_llc_dataset(n_examples, seed=0):
    rng = random.Random(seed)
    texts = []
    for _ in range(n_examples):
        topo = rng.choice(TOPOLOGIES)
        rung = rng.choice(RUNGS)
        inst = generate_instance(topo, rung, rng)
        pct = round(inst["oracle_answer"] * 100 / 5) * 5
        pct = max(0, min(100, pct))
        prompt = format_prompt(inst["question"])
        target = f" Answer: {pct}%"
        texts.append(prompt + target)
    return texts


def tokenize_dataset(model, texts, max_length=64):
    all_ids = []
    for text in texts:
        ids = model.to_tokens(text)[0].tolist()
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids = ids + [model.tokenizer.eos_token_id] * (max_length - len(ids))
        all_ids.append(ids)
    return Dataset.from_dict({"input_ids": all_ids}).with_format("torch")


def main():
    print("Loading tokenizer/model shell...")
    model = HookedTransformer.from_pretrained("gpt2").to(DEVICE)

    print("Building LLC sampling dataset...")
    texts = build_llc_dataset(N_LLC_EXAMPLES, seed=0)
    llc_ds = tokenize_dataset(model, texts)

    ckpt_files = sorted(
        f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ckpt_") and f.endswith(".pt")
    )
    steps = [int(f[len("ckpt_"):-len(".pt")]) for f in ckpt_files]
    print(f"Found {len(steps)} checkpoints: {steps}")

    llc_means, llc_stds = [], []
    for step, fname in zip(steps, ckpt_files):
        half_state = torch.load(f"{CHECKPOINT_DIR}/{fname}", map_location=DEVICE)
        full_state = {k: v.float() for k, v in half_state.items()}
        model.load_state_dict(full_state)
        model.eval()

        n_beta = default_nbeta(BATCH_SIZE)
        result = llc(
            model=model,
            dataset=llc_ds,
            observables={"train": llc_ds},
            lr=1e-5,
            n_beta=n_beta,
            num_chains=3,
            num_draws=50,
            batch_size=BATCH_SIZE,
            device=DEVICE,
        )
        mean = float(result["llc_mean"].values)
        std = float(result["llc_std"].values)
        llc_means.append(mean)
        llc_stds.append(std)
        print(f"Step {step}: LLC = {mean:.3f} +/- {std:.3f}")

        pickle.dump(
            {"step": steps[:len(llc_means)], "llc_mean": llc_means, "llc_std": llc_stds},
            open("results/phase_a2_llc.pkl", "wb"),
        )

    print("Saved results/phase_a2_llc.pkl")


if __name__ == "__main__":
    main()
