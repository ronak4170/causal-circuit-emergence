"""
Step 3.3: the single Phase A artifact -- per-rung accuracy (smoothed) and LLC
vs. training iteration, with detected jump points marked.

Jump detection criterion (defined here, before reading off conclusions from
the smoothed curves below -- see the honesty note in
results/phase_a_conclusion.md about why this isn't a strictly blind
pre-registration, since the raw per-iteration numbers were already watched
live during training):
  1. Smooth each rung's raw per-iteration accuracy (which is quantized to
     {0, .25, .5, .75, 1.0} because each iteration only samples 4 questions
     per rung) with a rolling mean, window = 10 iterations.
  2. A "jump" is the iteration index where the smoothed curve first increases
     by > 0.20 (20 percentage points) relative to its value 10 iterations
     earlier, and stays above the pre-jump level for the remainder of training
     (no jump is counted if the increase is transient).
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np

log = pickle.load(open("results/phase_a_log.pkl", "rb"))
llc = pickle.load(open("results/phase_a_llc.pkl", "rb"))

ITERS = np.array(log["iteration"])
WINDOW = 10


def smooth(x):
    """Rolling mean using only real data in each window (no zero-padding at
    the boundary -- np.convolve(..., mode='same') zero-pads, which creates a
    fake dip-then-rise artifact in the first WINDOW iterations that has
    nothing to do with real training dynamics)."""
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
    """First iteration where smoothed[i] - smoothed[i-window] > threshold AND
    the level is sustained (mean of the remaining tail stays above the
    pre-jump level) -- returns None if no such point exists."""
    for i in range(window, len(smoothed)):
        pre = smoothed[i - window]
        if smoothed[i] - pre > threshold:
            tail_mean = np.mean(smoothed[i:])
            if tail_mean > pre + threshold * 0.5:  # sustained, not a transient blip
                return iters[i]
    return None


jump_assoc = detect_jump(assoc_s, ITERS)
jump_interv = detect_jump(interv_s, ITERS)
jump_cf = detect_jump(cf_s, ITERS)

print(f"Detected jumps -- association: {jump_assoc}, intervention: {jump_interv}, "
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
ax1.set_title("Phase A: per-rung accuracy vs. training iteration\n"
               "(faint = raw per-iteration, bold = 10-iter rolling mean, dashed = detected jump)")
ax1.legend(loc="lower right")
ax1.set_ylim(-0.05, 1.05)

ax2.errorbar(llc["step"], llc["llc_mean"], yerr=llc["llc_std"], marker="o",
             color="tab:purple", capsize=3)
ax2.set_xlabel("Training iteration")
ax2.set_ylabel("Estimated LLC")
ax2.set_title("LLC vs. training iteration (10 checkpoints, RFT curriculum training)")

plt.tight_layout()
plt.savefig("results/phase_a_plot.png", dpi=150)
print("Saved results/phase_a_plot.png")

pickle.dump(
    {"jump_assoc": jump_assoc, "jump_interv": jump_interv, "jump_cf": jump_cf},
    open("results/phase_a_jumps.pkl", "wb"),
)
