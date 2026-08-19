"""
Phase F warm-up, RATIONALE condition: same association-only warm-up as the
original (warmup_sft.py, 300 examples instead of 400 to make room for the
new block) PLUS 100 intervention examples across all 4 topologies where the
target text includes a short worked do-calculus rationale (see
phase_f_rationale_templates.py) before the final "Answer: X%".

This is one of two warm-up conditions (see warmup_sft_intervention_control.py
for the other) that are IDENTICAL in every respect except whether the
intervention examples' target text includes the rationale -- isolating the
rationale's specific content as the only experimental variable, not just
"exposure to intervention-rung examples during warm-up" (which the original
warm-up had zero of).
"""
import os
import random

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance, QA_CONFIG
from rft_training_loop import format_prompt
from phase_f_rationale_templates import build_intervention_rationale

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ASSOC_EXAMPLES = 300
N_INTERV_EXAMPLES = 100
N_EPOCHS = 3
LR = 1e-5
TOPOLOGIES = ["chain", "fork", "collider", "confounded"]


def build_assoc_examples(n, seed=0):
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        topo = rng.choice(TOPOLOGIES)
        inst = generate_instance(topo, "association", rng)
        pct = round(inst["oracle_answer"] * 100 / 5) * 5
        pct = max(0, min(100, pct))
        prompt = format_prompt(inst["question"])
        target = f" Answer: {pct}%"
        examples.append((prompt, target))
    return examples


def main():
    os.makedirs("results", exist_ok=True)
    print("Building warm-up examples (RATIONALE condition)...")
    assoc_examples = build_assoc_examples(N_ASSOC_EXAMPLES, seed=0)

    # Build intervention examples directly (need do_value for the rationale text,
    # which generate_instance doesn't expose), mirroring generate_instance's own
    # intervention branch construction.
    from causal_dag_task import sample_structural_equations, estimate_intervention, render_intervention_question
    rng = random.Random(1)
    interv_examples = []
    for _ in range(N_INTERV_EXAMPLES):
        topo = rng.choice(TOPOLOGIES)
        cfg = QA_CONFIG[topo]
        do_var, target_var = cfg["treat_var"], cfg["target_var"]
        do_value = rng.random() < 0.5
        eqs, _ = sample_structural_equations(topo, rng)
        answer = estimate_intervention(eqs, target_var, do_var, do_value,
                                        n_samples=4000, seed=rng.randrange(1 << 30))
        question = render_intervention_question(do_var, do_value, target_var)
        pct = round(answer * 100 / 5) * 5
        pct = max(0, min(100, pct))
        rationale = build_intervention_rationale(topo, do_var, do_value, target_var)
        prompt = format_prompt(question)
        target = f" {rationale} Answer: {pct}%"
        interv_examples.append((prompt, target))

    examples = assoc_examples + interv_examples
    print(f"Built {len(assoc_examples)} association + {len(interv_examples)} "
          f"intervention-rationale examples ({len(examples)} total)")

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
    torch.save(model.state_dict(), "results/warmup_model_rationale.pt")
    print("Saved results/warmup_model_rationale.pt")

    from rft_training_loop import extract_predicted_probability
    check_rng = random.Random(999)
    n_parseable = 0
    n_rationale_present = 0
    n_check = 20
    for _ in range(n_check):
        topo = check_rng.choice(TOPOLOGIES)
        inst = generate_instance(topo, "intervention", check_rng)
        prompt = format_prompt(inst["question"])
        tokens = model.to_tokens(prompt)
        generated = model.generate(tokens, max_new_tokens=60, temperature=0.8,
                                    do_sample=True, verbose=False)
        new_tokens = generated[0, tokens.shape[1]:]
        gen_text = model.to_string(new_tokens)
        if extract_predicted_probability(gen_text) is not None:
            n_parseable += 1
        if len(gen_text.strip()) > 25:  # crude proxy: more than just "Answer: X%"
            n_rationale_present += 1
    print(f"Post-warmup format check (intervention prompts, max_new_tokens=60): "
          f"{n_parseable}/{n_check} parseable, {n_rationale_present}/{n_check} "
          f"produced more than a bare answer")


if __name__ == "__main__":
    main()
