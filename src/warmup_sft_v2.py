"""
RQ1 fix, part 1: a deliberately WEAKER warm-up than the original warmup_sft.py.

Phase A's original warm-up (400 examples, 3 epochs = 1200 total exposures)
solved reward sparsity but as a side effect saturated association accuracy to
~100% before RFT even began (see results/phase_a_conclusion.md) -- making
RQ1's "does association emerge first" claim untestable, since there was no
room left for it to emerge.

This version uses a much lighter touch (60 examples, 1 epoch = 60 total
exposures, a 20x reduction in warm-up dosage) -- enough to teach the output
FORMAT (a parseable "Answer: NN%" completion) so RFT isn't starting from zero
accepted rollouts, but deliberately not enough supervised exposure to drive
association accuracy to ceiling. The exact dosage was chosen by the Step
1.5-style checkpoint below: verify post-warmup association accuracy lands in
a genuine mid-range (not near-0%, not near-100%) before proceeding to RFT.
"""
import os
import random

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance
from rft_training_loop import format_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EXAMPLES = 60
N_EPOCHS = 1
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


def eval_association_accuracy(model, n=30, seed=888, tolerance=0.10):
    from rft_training_loop import extract_predicted_probability, is_correct
    rng = random.Random(seed)
    topologies = ["chain", "fork", "collider", "confounded"]
    n_correct, n_parseable = 0, 0
    for _ in range(n):
        topo = rng.choice(topologies)
        inst = generate_instance(topo, "association", rng)
        prompt = format_prompt(inst["question"])
        tokens = model.to_tokens(prompt)
        generated = model.generate(tokens, max_new_tokens=15, temperature=0.8,
                                    do_sample=True, verbose=False)
        new_tokens = generated[0, tokens.shape[1]:]
        gen_text = model.to_string(new_tokens)
        pred = extract_predicted_probability(gen_text)
        if pred is not None:
            n_parseable += 1
        if is_correct(pred, inst["oracle_answer"], tolerance):
            n_correct += 1
    return n_correct / n, n_parseable / n


def main():
    os.makedirs("results", exist_ok=True)
    print("Building weak warm-up examples...")
    examples = build_warmup_examples(N_EXAMPLES, seed=0)

    print("Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2").to(DEVICE)
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
    torch.save(model.state_dict(), "results/warmup_model_v2.pt")
    print("Saved results/warmup_model_v2.pt")

    acc, parseable_rate = eval_association_accuracy(model, n=30)
    print(f"Post-warmup association accuracy: {acc:.3f} (target: genuine mid-range, "
          f"NOT near-0 or near-1.0)")
    print(f"Post-warmup format-parseability rate: {parseable_rate:.3f} "
          f"(needs to be well above 0 for RFT to bootstrap)")
    if acc > 0.85:
        print("WARNING: association accuracy still near ceiling -- reduce N_EXAMPLES/"
              "N_EPOCHS further before using this checkpoint for Phase A v2.")
    elif parseable_rate < 0.3:
        print("WARNING: format-parseability too low -- RFT may struggle to bootstrap. "
              "Consider increasing N_EXAMPLES/N_EPOCHS slightly.")
    else:
        print("Warm-up dosage looks reasonable: format is learned, association has "
              "genuine room left to improve during RFT.")


if __name__ == "__main__":
    main()
