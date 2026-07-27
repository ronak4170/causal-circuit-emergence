"""
Step 4: cross-topology transfer test, adapted to how Step 3 was actually run
here.

Deviation from the starter code's exact design, disclosed: the starter code
assumes the candidate circuit was identified SEPARATELY per source topology,
then tests it on a different target topology in a full source x target
matrix. Step 3 here instead identified ONE candidate circuit from a MIXED
batch across all 4 topologies at once (60 pairs, 15/topology) -- there was no
per-topology circuit to build a source->target matrix from.

Adapted test: take that single candidate circuit (L7H5, L10H7, L8H11 heads;
L7/L9/L10/L11 MLP layers -- see results/phase_b_candidate_circuit.md) and
measure its patching effect SEPARATELY on each topology's own FRESH pairs
(new seed, not reused from Step 3, so these are genuinely held-out
questions even though the topology itself informed the original mixed-batch
identification). If the effect is consistently strong across all four
topologies, that supports a topology-general mechanism (RQ2 H2). If it's
strong on some and near-baseline on others, that's partial transfer. If it's
near-zero everywhere except whichever topology dominated the original 60-pair
batch, that falsifies topology-generality.
"""
import pickle

import torch

from phase_b_prompt_pairs import build_pair_batch, TOPOLOGY_VAR_MAP
from phase_b_patching import format_prompt, pct_token, logit_diff_metric
from phase_b_setup import load_model

CANDIDATE_HEADS = [(7, 5), (10, 7), (8, 11)]
CANDIDATE_MLP_LAYERS = [7, 9, 10, 11]


def patch_candidate_circuit_only(model, clean_cache, corrupt_tokens, heads, mlp_layers):
    hooks = []
    for (layer, head) in heads:
        def make_hook(layer=layer, head=head):
            def hook_fn(activation, hook):
                activation[:, -1, head, :] = clean_cache[f"blocks.{layer}.attn.hook_z"][:, -1, head, :]
                return activation
            return hook_fn
        hooks.append((f"blocks.{layer}.attn.hook_z", make_hook()))

    for layer in mlp_layers:
        def make_mlp_hook(layer=layer):
            def hook_fn(activation, hook):
                activation[:, -1, :] = clean_cache[f"blocks.{layer}.hook_mlp_out"][:, -1, :]
                return activation
            return hook_fn
        hooks.append((f"blocks.{layer}.hook_mlp_out", make_mlp_hook()))

    return model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)


def score_pair_with_circuit(model, pair, heads, mlp_layers):
    """Returns (baseline_score, circuit_patched_score), both normalized
    0 (unpatched corrupt) -> 1 (clean run), or None if the pair is unusable."""
    clean_prompt = format_prompt(pair["clean_prompt"]) + " Answer:"
    corrupt_prompt = format_prompt(pair["corrupt_prompt"]) + " Answer:"

    clean_tok = pct_token(model, pair["clean_answer"])
    corrupt_tok = pct_token(model, pair["corrupt_answer"])
    if clean_tok == corrupt_tok:
        return None

    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)
    if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
        return None

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupt_logits = model(corrupt_tokens)

    clean_score = logit_diff_metric(clean_logits, clean_tok, corrupt_tok)
    corrupt_score = logit_diff_metric(corrupt_logits, clean_tok, corrupt_tok)
    denom = clean_score - corrupt_score
    if abs(denom) < 1e-6:
        return None

    patched_logits = patch_candidate_circuit_only(model, clean_cache, corrupt_tokens,
                                                    heads, mlp_layers)
    patched_score = logit_diff_metric(patched_logits, clean_tok, corrupt_tok)
    normalized = (patched_score - corrupt_score) / denom
    return 0.0, normalized  # baseline is always 0 by construction of the normalization


def main():
    print("Loading checkpoint...")
    model = load_model()

    results = {}
    for topo in TOPOLOGY_VAR_MAP:
        pairs, _ = build_pair_batch(n_per_topology=15, seed=42)  # fresh seed, not Step 3's seed=0
        topo_pairs = [p for p in pairs if p["topology"] == topo]
        scores = []
        for pair in topo_pairs:
            result = score_pair_with_circuit(model, pair, CANDIDATE_HEADS, CANDIDATE_MLP_LAYERS)
            if result is not None:
                scores.append(result[1])
        mean_score = sum(scores) / len(scores) if scores else None
        results[topo] = {"scores": scores, "mean": mean_score, "n": len(scores)}
        print(f"{topo}: mean normalized restoration = {mean_score:.4f} (n={len(scores)})"
              if mean_score is not None else f"{topo}: no usable pairs")

    pickle.dump(results, open("results/phase_b_cross_topology_results.pkl", "wb"))
    print("Saved results/phase_b_cross_topology_results.pkl")


if __name__ == "__main__":
    main()
