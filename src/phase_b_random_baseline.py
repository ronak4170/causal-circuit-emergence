"""
Random-circuit baseline check for Phase B's cross-topology transfer result.

Gap being closed: the candidate circuit (L7H5, L10H7, L8H11 heads; MLP
layers 7/9/10/11) restored 90-98% of clean-run behavior across all 4
topologies. But this task is narrow and rigidly templated (a single
percentage token, teacher-forced position), so it's possible an ARBITRARY
set of 7 late-layer components would do nearly as well simply because so
much of the model's computation funnels into that one position regardless
of which specific components carry it. If that's true, "we found THE
circuit" is a much weaker claim than the transfer numbers alone suggest.

Method: sample N random circuits of the SAME SIZE AND COMPOSITION as the
candidate (3 attention heads + 4 MLP layers, drawn uniformly from all
144 heads / 12 layers), score each with the exact same cross-topology
transfer procedure used for the real candidate circuit, and compare the
resulting distribution against the candidate's actual score.
"""
import pickle
import random

import numpy as np

from phase_b_cross_topology import score_pair_with_circuit
from phase_b_prompt_pairs import build_pair_batch, TOPOLOGY_VAR_MAP
from phase_b_setup import load_model

N_RANDOM_CIRCUITS = 25
N_LAYERS, N_HEADS = 12, 12
CANDIDATE_HEADS = [(7, 5), (10, 7), (8, 11)]
CANDIDATE_MLP_LAYERS = [7, 9, 10, 11]


def sample_random_circuit(rng, n_heads=3, n_mlp=4):
    all_heads = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
    heads = rng.sample(all_heads, n_heads)
    mlp_layers = rng.sample(range(N_LAYERS), n_mlp)
    return heads, mlp_layers


def score_circuit_on_all_topologies(model, heads, mlp_layers, pairs_by_topology):
    per_topo_means = {}
    for topo, pairs in pairs_by_topology.items():
        scores = []
        for pair in pairs:
            result = score_pair_with_circuit(model, pair, heads, mlp_layers)
            if result is not None:
                scores.append(result[1])
        per_topo_means[topo] = sum(scores) / len(scores) if scores else None
    overall = np.mean([v for v in per_topo_means.values() if v is not None])
    return per_topo_means, overall


def main():
    print("Loading checkpoint...")
    model = load_model()

    print("Building fixed evaluation pairs (same seed=42 as the real transfer test)...")
    pairs, _ = build_pair_batch(n_per_topology=15, seed=42)
    pairs_by_topology = {topo: [p for p in pairs if p["topology"] == topo]
                          for topo in TOPOLOGY_VAR_MAP}

    print("\nScoring the REAL candidate circuit for reference...")
    candidate_per_topo, candidate_overall = score_circuit_on_all_topologies(
        model, CANDIDATE_HEADS, CANDIDATE_MLP_LAYERS, pairs_by_topology)
    print(f"Candidate circuit: {candidate_per_topo}, overall mean = {candidate_overall:.4f}")

    print(f"\nScoring {N_RANDOM_CIRCUITS} random circuits (3 heads + 4 MLP layers each)...")
    rng = random.Random(123)
    random_overalls = []
    random_details = []
    for i in range(N_RANDOM_CIRCUITS):
        heads, mlp_layers = sample_random_circuit(rng)
        per_topo, overall = score_circuit_on_all_topologies(model, heads, mlp_layers, pairs_by_topology)
        random_overalls.append(overall)
        random_details.append({"heads": heads, "mlp_layers": mlp_layers,
                                "per_topo": per_topo, "overall": overall})
        print(f"  [{i+1}/{N_RANDOM_CIRCUITS}] heads={heads} mlp={mlp_layers} "
              f"-> overall={overall:.4f}")

    random_overalls = np.array(random_overalls)
    percentile = (random_overalls < candidate_overall).mean() * 100

    print(f"\n--- Summary ---")
    print(f"Candidate circuit overall restoration: {candidate_overall:.4f}")
    print(f"Random circuits (n={N_RANDOM_CIRCUITS}): mean={random_overalls.mean():.4f}, "
          f"std={random_overalls.std():.4f}, min={random_overalls.min():.4f}, "
          f"max={random_overalls.max():.4f}")
    print(f"Candidate circuit is at the {percentile:.1f}th percentile of the random distribution")

    pickle.dump(
        {"candidate_per_topo": candidate_per_topo, "candidate_overall": candidate_overall,
         "random_overalls": random_overalls, "random_details": random_details,
         "percentile": percentile},
        open("results/phase_b_random_baseline.pkl", "wb"),
    )
    print("Saved results/phase_b_random_baseline.pkl")


if __name__ == "__main__":
    main()
