"""
Phase F warm-up, CONTROL condition: identical to warmup_sft_rationale.py in
every respect -- same association examples (same seed), same 100
intervention examples across all 4 topologies (same seed, same do_var/
do_value/topology draws) -- EXCEPT the intervention target text is plain
"Answer: X%", with no rationale. This isolates "exposure to intervention-
rung examples during warm-up" (which the ORIGINAL warmup_sft.py had zero of)
as a confound, so any behavioral difference between this control and the
rationale condition can be attributed specifically to the rationale's
content, not to extra intervention-format exposure.
"""
import os
import random

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance, QA_CONFIG
from causal_dag_task import sample_structural_equations, estimate_intervention, render_intervention_question
from rft_training_loop import format_prompt

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
    print("Building warm-up examples (CONTROL condition -- no rationale)...")
    assoc_examples = build_assoc_examples(N_ASSOC_EXAMPLES, seed=0)

    # Same seed=1 draw sequence as warmup_sft_rationale.py, so the SAME
    # topology/do_var/do_value/eqs are used -- only the target text format differs.
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
        prompt = format_prompt(question)
        target = f" Answer: {pct}%"
        interv_examples.append((prompt, target))

    examples = assoc_examples + interv_examples
    print(f"Built {len(assoc_examples)} association + {len(interv_examples)} "
          f"intervention (no-rationale) examples ({len(examples)} total)")

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
    torch.save(model.state_dict(), "results/warmup_model_intervention_control.pt")
    print("Saved results/warmup_model_intervention_control.pt")

    from rft_training_loop import extract_predicted_probability
    check_rng = random.Random(999)
    n_parseable = 0
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
    print(f"Post-warmup format check (intervention prompts, max_new_tokens=60): "
          f"{n_parseable}/{n_check} parseable")


if __name__ == "__main__":
    main()
