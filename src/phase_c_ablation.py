"""
Step 3: mean-ablate the Phase B candidate circuit and evaluate on the Phase C
test set. Per PREREGISTRATION.md (committed e2ae1c7 before this file existed):
mean ablation, not zero ablation.

Deviation from the starter code's evaluation loop, for reliability (the
starter code itself flagged this as an honesty risk -- "verify the exact
TransformerLens API for combining generate + fwd_hooks... test on ONE
question end-to-end before running the full test set"): rather than
multi-token .generate() with hooks, this uses a single forward pass with
greedy (argmax) decoding of the next token after "...Answer:", reusing the
same fact from Phase B that every 5%-rounded percentage 0-100 tokenizes as
ONE token in GPT-2's BPE. This is deterministic (matches the pre-registered
temperature=0.0 / greedy intent exactly), avoids the flagged API uncertainty
entirely, and is far cheaper (1 forward pass vs up to 15 generation steps).
"""
import random
import re

import torch
from transformer_lens import HookedTransformer

from causal_dag_task import generate_instance
from phase_b_setup import load_model
from rft_training_loop import format_prompt

CANDIDATE_HEADS = [(7, 5), (10, 7), (8, 11)]
CANDIDATE_MLP_LAYERS = [7, 9, 10, 11]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_diverse_prompts(n=100, seed=7):
    """Diverse causal-DAG prompts spanning all 3 rungs and 4 topologies, used
    to compute mean ablation targets (per PREREGISTRATION.md's disclosed
    deviation: a freshly-sampled equivalent distribution, since Phase A's
    literal training prompt log wasn't saved verbatim)."""
    rng = random.Random(seed)
    topologies = ["chain", "fork", "collider", "confounded"]
    rungs = ["association", "intervention", "counterfactual"]
    prompts = []
    for _ in range(n):
        topo = rng.choice(topologies)
        rung = rng.choice(rungs)
        inst = generate_instance(topo, rung, rng)
        prompts.append(format_prompt(inst["question"]) + " Answer:")
    return prompts


def compute_component_means(model, prompts):
    all_head_activations = {(l, h): [] for (l, h) in CANDIDATE_HEADS}
    all_mlp_activations = {l: [] for l in CANDIDATE_MLP_LAYERS}

    for prompt in prompts:
        tokens = model.to_tokens(prompt).to(DEVICE)
        _, cache = model.run_with_cache(tokens)
        for (l, h) in CANDIDATE_HEADS:
            act = cache[f"blocks.{l}.attn.hook_z"][:, :, h, :].mean(dim=(0, 1))
            all_head_activations[(l, h)].append(act)
        for l in CANDIDATE_MLP_LAYERS:
            act = cache[f"blocks.{l}.hook_mlp_out"][:, :, :].mean(dim=(0, 1))
            all_mlp_activations[l].append(act)

    head_means = {k: torch.stack(v).mean(dim=0) for k, v in all_head_activations.items()}
    mlp_means = {k: torch.stack(v).mean(dim=0) for k, v in all_mlp_activations.items()}
    return head_means, mlp_means


def build_ablation_hooks(head_means, mlp_means):
    hooks = []
    for (l, h), mean_act in head_means.items():
        def make_head_hook(l=l, h=h, mean_act=mean_act):
            def hook_fn(activation, hook):
                activation[:, :, h, :] = mean_act
                return activation
            return hook_fn
        hooks.append((f"blocks.{l}.attn.hook_z", make_head_hook()))

    for l, mean_act in mlp_means.items():
        def make_mlp_hook(l=l, mean_act=mean_act):
            def hook_fn(activation, hook):
                activation[:, :, :] = mean_act
                return activation
            return hook_fn
        hooks.append((f"blocks.{l}.hook_mlp_out", make_mlp_hook()))

    return hooks


def predicted_percentage_greedy(model, prompt, fwd_hooks=None):
    """Single forward pass, greedy next-token after '...Answer:', decoded as
    a percentage. Returns None if the argmax token isn't a valid 0-100 number."""
    tokens = model.to_tokens(prompt).to(DEVICE)
    if fwd_hooks:
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    else:
        logits = model(tokens)
    next_token_id = logits[0, -1, :].argmax().item()
    token_str = model.tokenizer.decode([next_token_id])
    match = re.search(r"(\d{1,3})", token_str)
    if match:
        val = int(match.group(1))
        if 0 <= val <= 100:
            return val / 100.0
    return None


def evaluate(model, test_questions, fwd_hooks=None):
    results = []
    for q in test_questions:
        prompt = format_prompt(q["prompt"]) + " Answer:"
        pred = predicted_percentage_greedy(model, prompt, fwd_hooks=fwd_hooks)
        entry = {"topology": q["topology"], "prompt": q["prompt"], "predicted": pred}
        if "interventional_answer" in q:
            entry["interventional_true"] = q["interventional_answer"]
            entry["associational_true"] = q["associational_answer"]
        else:
            entry["associational_true"] = q["associational_answer"]
        results.append(entry)
    return results
