"""
Phase E evaluation: for each (condition, generation) checkpoint, compute:
1. Discrimination-test correlation on chain/fork/collider (Phase D's method
   -- verified genuine causal competence, the metric Prediction 1 tracks).
2. Confounded-topology floor check (Phase C's method -- still 0%? changed?).
3. Output diversity (std. dev. of repeated-sample answers at temperature 0.8,
   the standard collapse metric).
4. Qualitative ablation-under-mean-ablation check on confounded (does Phase
   C's constant-collapse pattern persist, per the "also tracked, not
   pre-registered" item in PREREGISTRATION.md).
"""
import os
import pickle

import torch
from transformer_lens import HookedTransformer

from phase_b_setup import load_model as load_base_model  # for the shell/tokenizer
from phase_c_ablation import (build_diverse_prompts, compute_component_means,
                                build_ablation_hooks, predicted_percentage_greedy,
                                CANDIDATE_HEADS, CANDIDATE_MLP_LAYERS)
from phase_d_discrimination_test import build_discrimination_test_set, pearson_corr
from phase_c_reference_answers import build_phase_c_test_set
from rft_training_loop import format_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = ["A", "B", "C", "D"]
N_GENERATIONS = 3


def load_checkpoint_model(condition, generation):
    model = HookedTransformer.from_pretrained("gpt2").to(DEVICE)
    half_state = torch.load(
        f"results/phase_e_checkpoints/condition_{condition}_gen_{generation}.pt",
        map_location=DEVICE)
    full_state = {k: v.float() for k, v in half_state.items()}
    model.load_state_dict(full_state)
    model.eval()
    return model


def discrimination_correlation(model, test_set):
    """Metric 1: per-topology correlation between predicted and true answers."""
    from collections import defaultdict
    by_topo = defaultdict(list)
    for q in test_set:
        prompt = format_prompt(q["prompt"]) + " Answer:"
        pred = predicted_percentage_greedy(model, prompt)
        if pred is not None:
            by_topo[q["topology"]].append((pred, q["true_answer"]))

    corrs = {}
    for topo, pairs in by_topo.items():
        preds, trues = zip(*pairs)
        corrs[topo] = pearson_corr(list(preds), list(trues))
    return corrs


def confounded_floor_check(model, n=30, seed=555):
    """Metric 2: is confounded-topology accuracy still ~0%?"""
    import random
    from causal_dag_task import generate_instance
    rng = random.Random(seed)
    n_correct = 0
    for _ in range(n):
        inst = generate_instance("confounded", "intervention", rng)
        prompt = format_prompt(inst["question"]) + " Answer:"
        pred = predicted_percentage_greedy(model, prompt)
        if pred is not None and abs(pred - inst["oracle_answer"]) <= 0.10:
            n_correct += 1
    return n_correct / n


def output_diversity(model, n_repeats=10, seed=777):
    """Metric 3: std. dev. of the model's generated answer across n_repeats
    samples of the SAME prompt at temperature 0.8 -- shrinking std = collapse."""
    import random
    from causal_dag_task import generate_instance
    from rft_training_loop import extract_predicted_probability

    rng = random.Random(seed)
    inst = generate_instance("chain", "intervention", rng)
    prompt = format_prompt(inst["question"]) + " Answer:"
    tokens = model.to_tokens(prompt)

    preds = []
    for _ in range(n_repeats):
        generated = model.generate(tokens, max_new_tokens=8, temperature=0.8,
                                    do_sample=True, verbose=False)
        new_tokens = generated[0, tokens.shape[1]:]
        gen_text = model.to_string(new_tokens)
        pred = extract_predicted_probability(gen_text)
        if pred is not None:
            preds.append(pred)

    if len(preds) < 2:
        return None
    mean = sum(preds) / len(preds)
    var = sum((p - mean) ** 2 for p in preds) / len(preds)
    return var ** 0.5


def ablation_qualitative_check(model, n_pairs=20, seed=888):
    """Metric 4: does confounded's mean-ablation constant-collapse pattern
    (Phase C's finding) still hold? Returns the set of distinct output values
    and whether it's still a single constant."""
    import random
    diverse_prompts = build_diverse_prompts(n=50, seed=seed)
    head_means, mlp_means = compute_component_means(model, diverse_prompts)
    hooks = build_ablation_hooks(head_means, mlp_means)

    from causal_dag_task import generate_instance
    rng = random.Random(seed)
    preds = []
    for _ in range(n_pairs):
        inst = generate_instance("confounded", "intervention", rng)
        prompt = format_prompt(inst["question"]) + " Answer:"
        pred = predicted_percentage_greedy(model, prompt, fwd_hooks=hooks)
        preds.append(pred)

    distinct = set(p for p in preds if p is not None)
    return {"distinct_values": distinct, "n_distinct": len(distinct), "is_constant": len(distinct) <= 1}


def main():
    print("Building shared test sets...")
    discrimination_test_set = build_discrimination_test_set(n_per_topology=20, seed=99)

    results = {}
    for condition in CONDITIONS:
        results[condition] = {}
        for gen in range(0, N_GENERATIONS + 1):
            path = f"results/phase_e_checkpoints/condition_{condition}_gen_{gen}.pt"
            if not os.path.exists(path):
                print(f"Skipping condition {condition} gen {gen}: checkpoint not found")
                continue
            model = load_checkpoint_model(condition, gen)

            corrs = discrimination_correlation(model, discrimination_test_set)
            floor = confounded_floor_check(model)
            diversity = output_diversity(model)
            ablation_check = ablation_qualitative_check(model)

            results[condition][gen] = {
                "discrimination_corr": corrs, "confounded_accuracy": floor,
                "output_diversity_std": diversity, "ablation_check": ablation_check,
            }
            print(f"Condition {condition} gen {gen}: corr={corrs}, "
                  f"confounded_acc={floor:.2f}, diversity_std={diversity}, "
                  f"ablation_distinct={ablation_check['n_distinct']}")

    pickle.dump(results, open("results/phase_e_evaluations.pkl", "wb"))
    print("\nSaved results/phase_e_evaluations.pkl")


if __name__ == "__main__":
    main()
