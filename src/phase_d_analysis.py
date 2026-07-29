"""
Step 5: analyze the implicit/explicit gap trajectory across all 10 Phase A
checkpoints, with Wilson-score confidence intervals (n=15 per topology per
checkpoint is small, so a normal approximation would be unreliable near 0/1
-- Wilson handles that correctly without an extra dependency).
"""
import math
import pickle

import matplotlib.pyplot as plt

TOPOLOGIES = ["chain", "fork", "collider", "confounded"]
N_PER_TOPOLOGY = 15


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (None, None)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def cis_overlap(ci1, ci2):
    return not (ci1[1] < ci2[0] or ci2[1] < ci1[0])


def main():
    results = pickle.load(open("results/phase_d_results.pkl", "rb"))
    steps = sorted(results.keys())

    print("Per-checkpoint, per-topology gap with Wilson CI overlap check:\n")
    analysis = {topo: {"steps": [], "implicit": [], "explicit": [], "gap": [],
                        "implicit_ci": [], "explicit_ci": [], "gap_closed": []}
                for topo in TOPOLOGIES}

    for step in steps:
        r = results[step]
        for topo in TOPOLOGIES:
            i_acc = r["implicit"][topo]
            e_acc = r["explicit"][topo]
            k_i = round(i_acc * N_PER_TOPOLOGY)
            k_e = round(e_acc * N_PER_TOPOLOGY)
            ci_i = wilson_ci(k_i, N_PER_TOPOLOGY)
            ci_e = wilson_ci(k_e, N_PER_TOPOLOGY)
            closed = cis_overlap(ci_i, ci_e)

            analysis[topo]["steps"].append(step)
            analysis[topo]["implicit"].append(i_acc)
            analysis[topo]["explicit"].append(e_acc)
            analysis[topo]["gap"].append(e_acc - i_acc)
            analysis[topo]["implicit_ci"].append(ci_i)
            analysis[topo]["explicit_ci"].append(ci_e)
            analysis[topo]["gap_closed"].append(closed)

            print(f"  step={step:4d} {topo:11s} implicit={i_acc:.2f} {ci_i[0]:.2f}-{ci_i[1]:.2f}  "
                  f"explicit={e_acc:.2f} {ci_e[0]:.2f}-{ci_e[1]:.2f}  "
                  f"gap={e_acc-i_acc:+.2f}  CI-overlap(closed)={closed}")

    # ---- Plot: gap vs step per topology ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, topo in zip(axes.flat, TOPOLOGIES):
        d = analysis[topo]
        ax.plot(d["steps"], d["gap"], marker="o", color="tab:purple")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(topo)
        ax.set_ylim(-0.6, 0.6)
    for ax in axes[-1, :]:
        ax.set_xlabel("Training iteration")
    for ax in axes[:, 0]:
        ax.set_ylabel("Explicit - Implicit accuracy (gap)")
    fig.suptitle("Phase D: implicit/explicit accuracy gap vs. training iteration, per topology")
    plt.tight_layout()
    plt.savefig("results/phase_d_gap_plot.png", dpi=150)
    print("\nSaved results/phase_d_gap_plot.png")

    pickle.dump(analysis, open("results/phase_d_analysis.pkl", "wb"))
    print("Saved results/phase_d_analysis.pkl")


if __name__ == "__main__":
    main()
