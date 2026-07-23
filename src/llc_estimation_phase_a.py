"""
LLC estimation over Phase A checkpoints (Step 3.2). Reuses the exact same
devinterp API validated in Phase 0 (devinterp.slt.llc.llc(), same call
pattern as devinterp's own examples/quickstart.py) rather than writing new
LLC code from scratch.

Unlike the Phase 0 grokking task, this uses devinterp's DEFAULT per-token
cross-entropy loss_fn (no custom loss_fn needed): the "training data" for
the LLC's internal SGLD sampler is a set of standard (prompt + correct
answer) causal-LM sequences -- exactly the same form finetune_on_accepted()
already trains on -- so the default next-token loss is the right measure of
this model's local loss landscape.

Checkpoints were saved in float16 (see phase_a_main.py's disk-space fix);
they are cast back to float32 on load since SGLD sampling needs float32.
"""
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
# MPS is intentionally excluded from auto-detection: TransformerLens warns it "may
# produce silently incorrect results" (see docs/phase0_setup.md). CUDA has no such
# documented issue, so it's used automatically when available (e.g. on Colab).
CHECKPOINT_DIR = "results/phase_a_checkpoints"
BATCH_SIZE = 16
N_LLC_EXAMPLES = 256
TOPOLOGIES = ["chain", "fork", "collider", "confounded"]
RUNGS = ["association", "intervention", "counterfactual"]


def build_llc_dataset(n_examples, seed=0):
    """Same style as warmup_sft.build_warmup_examples but spanning all three
    rungs, matching what the model was actually RFT-trained on."""
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

    print("Building LLC sampling dataset (same distribution the model was RFT-trained on)...")
    texts = build_llc_dataset(N_LLC_EXAMPLES, seed=0)
    llc_ds = tokenize_dataset(model, texts)

    import os
    ckpt_files = sorted(
        f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ckpt_") and f.endswith(".pt")
    )
    steps = [int(f[len("ckpt_"):-len(".pt")]) for f in ckpt_files]

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
        {"step": steps, "llc_mean": llc_means, "llc_std": llc_stds},
        open("results/phase_a_llc.pkl", "wb"),
    )
    print("Saved results/phase_a_llc.pkl")


if __name__ == "__main__":
    main()
