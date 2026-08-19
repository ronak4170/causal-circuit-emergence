"""
Pythia-410M version of phase_b_patching.py. The patching logic itself
(patch_and_score) is already model-agnostic -- it reads n_layers/n_heads
from model.cfg dynamically -- so this file only swaps the setup import and
output paths. Pythia has 24 layers x 16 heads = 384 total heads (vs GPT-2's
144), so this will take proportionally longer per pair.
"""
import pickle

from phase_b_patching import patch_and_score
from phase_b_prompt_pairs import build_pair_batch
from phase_b_setup_pythia import load_model, POST_TRANSITION_STEP


def main():
    print(f"Loading Pythia checkpoint at step {POST_TRANSITION_STEP}...")
    model = load_model()
    print(f"Config: n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads} "
          f"({model.cfg.n_layers * model.cfg.n_heads} total heads)")

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

    print(f"Done patching. {len(all_head_results)} usable pairs, {n_skipped} skipped.")

    import torch
    avg_head_results = torch.stack(all_head_results).mean(dim=0)
    avg_mlp_results = torch.stack(all_mlp_results).mean(dim=0)

    pickle.dump(
        {"avg_head_results": avg_head_results, "avg_mlp_results": avg_mlp_results,
         "n_pairs_used": len(all_head_results)},
        open("results/phase_b_pythia_patching_results.pkl", "wb"),
    )

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_head_results.detach().cpu().numpy(), cmap="RdBu", aspect="auto")
    plt.colorbar(label="Avg normalized shift toward clean answer")
    plt.xlabel("Head"); plt.ylabel("Layer")
    plt.title(f"Pythia-410M: attention heads implicated in interventional reasoning\n"
              f"(averaged across {len(all_head_results)} pairs, 4 topologies)")
    plt.savefig("results/phase_b_pythia_head_heatmap.png", dpi=150)

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(avg_mlp_results)), avg_mlp_results.detach().cpu().numpy())
    plt.xlabel("Layer"); plt.ylabel("Avg normalized shift toward clean answer")
    plt.title("Pythia-410M: MLP layers implicated in interventional reasoning")
    plt.tight_layout()
    plt.savefig("results/phase_b_pythia_mlp_barplot.png", dpi=150)

    print("Saved results/phase_b_pythia_patching_results.pkl, "
          "results/phase_b_pythia_head_heatmap.png, results/phase_b_pythia_mlp_barplot.png")


if __name__ == "__main__":
    main()
