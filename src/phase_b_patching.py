"""
Step 3: activation patching across every attention head (hook_z) and every
MLP layer (hook_mlp_out) to find which components are causally responsible
for interventional-reasoning competence.

Metric design (the get_predicted_prob placeholder from the starter code,
made concrete): GPT-2's BPE tokenizes every 5%-rounded percentage 0-100 as a
SINGLE token with a leading space (verified directly: e.g. " 85" -> one
token). The model's own training format is "...Answer: {pct}%", and
"...Answer:" tokenizes as [' Answer', ':'] -- so teacher-forcing the prompt
up through "Answer:" and reading the next-token logit at that final position
gives a clean single-forward-pass metric: logit[clean_pct_token] -
logit[corrupt_pct_token]. This mirrors the IOI paper's single-token
logit-diff metric exactly, and (crucially for cost) needs only ONE forward
pass per patch -- no autoregressive generation required, unlike Phase A's
free-form-percentage scoring.

The metric is normalized 0 (matches unpatched corrupt run) -> 1 (matches
clean run) so patching effects are comparable across pairs with different
raw logit-diff scales.
"""
import pickle

import torch
from transformer_lens import HookedTransformer

from phase_b_prompt_pairs import build_pair_batch
from phase_b_setup import load_model, POST_TRANSITION_STEP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def format_prompt(question):
    return f"Question: {question}\nAnswer (as a percentage, e.g. 'Answer: 42%'):"


def pct_token(model, pct):
    pct = max(0, min(100, round(pct * 100 / 5) * 5))
    toks = model.to_tokens(f" {pct}", prepend_bos=False)[0]
    return toks[0].item()


def logit_diff_metric(logits, clean_tok, corrupt_tok):
    final = logits[0, -1, :]
    return (final[clean_tok] - final[corrupt_tok]).item()


def patch_and_score(model, pair):
    """For one clean/corrupt pair, patch every attention head and every MLP
    layer, scoring each patch's effect on the normalized logit-diff metric."""
    clean_prompt = format_prompt(pair["clean_prompt"]) + " Answer:"
    corrupt_prompt = format_prompt(pair["corrupt_prompt"]) + " Answer:"

    clean_tok = pct_token(model, pair["clean_answer"])
    corrupt_tok = pct_token(model, pair["corrupt_answer"])
    if clean_tok == corrupt_tok:
        return None  # rounded to the same token -- metric undefined, skip

    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)
    if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
        return None  # minimal-pair assumption (same length) violated, skip

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupt_logits = model(corrupt_tokens)

    clean_score = logit_diff_metric(clean_logits, clean_tok, corrupt_tok)
    corrupt_score = logit_diff_metric(corrupt_logits, clean_tok, corrupt_tok)
    denom = clean_score - corrupt_score
    if abs(denom) < 1e-6:
        return None

    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    head_results = torch.zeros(n_layers, n_heads)
    mlp_results = torch.zeros(n_layers)

    for layer in range(n_layers):
        for head in range(n_heads):
            def patch_head(activation, hook, layer=layer, head=head):
                activation[:, -1, head, :] = clean_cache[f"blocks.{layer}.attn.hook_z"][:, -1, head, :]
                return activation
            patched_logits = model.run_with_hooks(
                corrupt_tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_z", patch_head)])
            score = logit_diff_metric(patched_logits, clean_tok, corrupt_tok)
            head_results[layer, head] = (score - corrupt_score) / denom

        def patch_mlp(activation, hook, layer=layer):
            activation[:, -1, :] = clean_cache[f"blocks.{layer}.hook_mlp_out"][:, -1, :]
            return activation
        patched_logits_mlp = model.run_with_hooks(
            corrupt_tokens, fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", patch_mlp)])
        score = logit_diff_metric(patched_logits_mlp, clean_tok, corrupt_tok)
        mlp_results[layer] = (score - corrupt_score) / denom

    return head_results, mlp_results


def main():
    print(f"Loading checkpoint at step {POST_TRANSITION_STEP}...")
    model = load_model()

    print("Building clean/corrupt pairs across all 4 topologies...")
    pairs, counts = build_pair_batch(n_per_topology=15, seed=0)
    print(f"Pair counts: {counts}")

    all_head_results, all_mlp_results = [], []
    n_skipped = 0
    for i, pair in enumerate(pairs):
        result = patch_and_score(model, pair)
        if result is None:
            n_skipped += 1
            continue
        head_res, mlp_res = result
        all_head_results.append(head_res)
        all_mlp_results.append(mlp_res)
        print(f"  [{i+1}/{len(pairs)}] {pair['topology']}: patched "
              f"({len(all_head_results)} usable so far, {n_skipped} skipped)")

    print(f"Done patching. {len(all_head_results)} usable pairs, {n_skipped} skipped "
          f"(token collision or length mismatch).")

    avg_head_results = torch.stack(all_head_results).mean(dim=0)
    avg_mlp_results = torch.stack(all_mlp_results).mean(dim=0)

    pickle.dump(
        {"avg_head_results": avg_head_results, "avg_mlp_results": avg_mlp_results,
         "n_pairs_used": len(all_head_results)},
        open("results/phase_b_patching_results.pkl", "wb"),
    )

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.imshow(avg_head_results.detach().cpu().numpy(), cmap="RdBu", aspect="auto")
    plt.colorbar(label="Avg normalized shift toward clean answer")
    plt.xlabel("Head"); plt.ylabel("Layer")
    plt.title(f"Attention heads implicated in interventional reasoning\n"
              f"(averaged across {len(all_head_results)} pairs, 4 topologies)")
    plt.savefig("results/phase_b_head_heatmap.png", dpi=150)

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(avg_mlp_results)), avg_mlp_results.detach().cpu().numpy())
    plt.xlabel("Layer"); plt.ylabel("Avg normalized shift toward clean answer")
    plt.title("MLP layers implicated in interventional reasoning")
    plt.tight_layout()
    plt.savefig("results/phase_b_mlp_barplot.png", dpi=150)

    print("Saved results/phase_b_patching_results.pkl, "
          "results/phase_b_head_heatmap.png, results/phase_b_mlp_barplot.png")


if __name__ == "__main__":
    main()
