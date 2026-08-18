"""
Pythia-410M version of warmup_sft.py -- the Phase B/C/D scale-up replication
(publication roadmap experiment #1: "reproduce consistency-vs-correctness on
Pythia-410M"). Uses the ORIGINAL warm-up strength (400 examples, 3 epochs),
not the weakened v2 version -- this replication is about Phase B/C/D's
headline finding, not RQ1's emergence story, so maximum comparability to the
original GPT-2 Phase A/B/C/D results is the goal here.
"""
import os
import random

import torch
import torch.nn as nn

import pythia_compat  # noqa: F401 -- side-effecting import, must come before HookedTransformer
from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance
from rft_training_loop import format_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "EleutherAI/pythia-410m"
N_EXAMPLES = 400
N_EPOCHS = 3
LR = 1e-5


def build_warmup_examples(n, seed=0):
    rng = random.Random(seed)
    topologies = ["chain", "fork", "collider", "confounded"]
    examples = []
    for _ in range(n):
        topo = rng.choice(topologies)
        inst = generate_instance(topo, "association", rng)
        pct = round(inst["oracle_answer"] * 100 / 5) * 5
        pct = max(0, min(100, pct))
        prompt = format_prompt(inst["question"])
        target = f" Answer: {pct}%"
        examples.append((prompt, target))
    return examples


def main():
    os.makedirs("results", exist_ok=True)
    print("Building warm-up examples...")
    examples = build_warmup_examples(N_EXAMPLES, seed=0)

    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME).to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        random.Random(epoch).shuffle(examples)
        total_loss = 0.0
        for prompt, target in examples:
            full_text = prompt + target
            tokens = model.to_tokens(full_text).to(DEVICE)
            logits = model(tokens)
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                tokens[:, 1:].reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: mean loss = {total_loss / len(examples):.4f}")

    model.eval()
    torch.save(model.state_dict(), "results/warmup_model_pythia.pt")
    print("Saved results/warmup_model_pythia.pt")

    from rft_training_loop import extract_predicted_probability
    rng = random.Random(999)
    n_parseable = 0
    n_check = 20
    for _ in range(n_check):
        inst = generate_instance(random.choice(["chain", "fork", "collider", "confounded"]),
                                  "association", rng)
        prompt = format_prompt(inst["question"])
        tokens = model.to_tokens(prompt)
        generated = model.generate(tokens, max_new_tokens=15, temperature=0.8,
                                    do_sample=True, verbose=False)
        new_tokens = generated[0, tokens.shape[1]:]
        gen_text = model.to_string(new_tokens)
        if extract_predicted_probability(gen_text) is not None:
            n_parseable += 1
    print(f"Post-warmup format check: {n_parseable}/{n_check} generations were parseable percentages")


if __name__ == "__main__":
    main()
