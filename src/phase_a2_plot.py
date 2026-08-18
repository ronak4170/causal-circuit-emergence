"""
RQ1-fix analysis: same jump-detection method as phase_a_plot.py (identical
window/threshold, for direct comparability with the original Phase A run),
applied to the v2 (weakened warm-up + doubled batch) log and LLC data.
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np

log = pickle.load(open("results/phase_a2_log.pkl", "rb"))
llc = pickle.load(open("results/phase_a2_llc.pkl", "rb"))

ITERS = np.array(log["iteration"])
WINDOW = 10


def smooth(x):
    x = np.array(x, dtype=float)
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - WINDOW + 1)
        out[i] = x[lo:i + 1].mean()
    return out


assoc_s = smooth(log["assoc_acc"])
interv_s = smooth(log["interv_acc"])
cf_s = smooth(log["cf_acc"])


def detect_jump(smoothed, iters, window=WINDOW, threshold=0.20):
    for i in range(window, len(smoothed)):
        pre = smoothed[i - window]
        if smoothed[i] - pre > threshold:
            tail_mean = np.mean(smoothed[i:])
            if tail_mean > pre + threshold * 0.5:
                return iters[i]
    return None


jump_assoc = detect_jump(assoc_s, ITERS)
jump_interv = detect_jump(interv_s, ITERS)
jump_cf = detect_jump(cf_s, ITERS)

print(f"Detected jumps (v2) -- association: {jump_assoc}, intervention: {jump_interv}, "
      f"counterfactual: {jump_cf}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

ax1.plot(ITERS, log["assoc_acc"], color="tab:blue", alpha=0.2)
ax1.plot(ITERS, log["interv_acc"], color="tab:orange", alpha=0.2)
ax1.plot(ITERS, log["cf_acc"], color="tab:green", alpha=0.2)
ax1.plot(ITERS, assoc_s, color="tab:blue", label="association (smoothed)", linewidth=2)
ax1.plot(ITERS, interv_s, color="tab:orange", label="intervention (smoothed)", linewidth=2)
ax1.plot(ITERS, cf_s, color="tab:green", label="counterfactual (smoothed)", linewidth=2)

for jump, color in [(jump_assoc, "tab:blue"), (jump_interv, "tab:orange"), (jump_cf, "tab:green")]:
    if jump is not None:
        ax1.axvline(jump, color=color, linestyle="--", alpha=0.7)

ax1.set_ylabel("Accuracy (within tolerance)")
ax1.set_title("Phase A v2 (RQ1 fix): per-rung accuracy vs. training iteration\n"
               "(faint = raw per-iteration, bold = 10-iter rolling mean, dashed = detected jump)")
ax1.legend(loc="lower right")
ax1.set_ylim(-0.05, 1.05)

ax2.errorbar(llc["step"], llc["llc_mean"], yerr=llc["llc_std"], marker="o",
             color="tab:purple", capsize=3)
ax2.set_xlabel("Training iteration")
ax2.set_ylabel("Estimated LLC")
ax2.set_title("LLC vs. training iteration (v2: 16 checkpoints, weakened warm-up)")

plt.tight_layout()
plt.savefig("results/phase_a2_plot.png", dpi=150)
print("Saved results/phase_a2_plot.png")

pickle.dump(
    {"jump_assoc": jump_assoc, "jump_interv": jump_interv, "jump_cf": jump_cf},
    open("results/phase_a2_jumps.pkl", "wb"),
)
