# Phase 0: Getting Started — Environment Setup & Foundational Reproductions

**Goal:** Before touching the causal-reasoning research question, get working tools and proof you can use them correctly. This phase reproduces two already-known results (a grokking phase transition, and a known circuit) so that when you later find something in your own experiment, you trust your methodology.

**Do not skip ahead to your own research question yet.** If you can't reproduce a known result, you can't trust a novel one.

**Time budget:** 2–3 weeks. **Compute:** single GPU; the grokking part can run on CPU if GPU access isn't ready.

---

## Step 1: Environment Setup

```bash
python3 -m venv causal_interp_env
source causal_interp_env/bin/activate   # Windows: causal_interp_env\Scripts\activate
pip install --upgrade pip

# CUDA version must match your cluster — check with: nvidia-smi
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformer_lens devinterp
pip install numpy pandas matplotlib seaborn jupyter tqdm scikit-learn networkx
```

Verify GPU access:
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```
If this doesn't show your university GPU, fix that with HPC support before anything else.

Set up the project and use git from day one (matters later for pre-registering your ablation prediction with a dated commit):
```bash
mkdir -p causal_circuits_project/{notebooks,src,results,data,logs}
cd causal_circuits_project
git init
```

---

## Step 2: Reproduction Exercise A — Grokking + Local Learning Coefficient

**Task:** train a tiny transformer on modular addition `(a+b) mod p`, the canonical grokking task (Power et al. 2022; Nanda et al. 2023).

```python
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer, HookedTransformerConfig

P = 113
D_MODEL, N_LAYERS, N_HEADS, D_HEAD, SEED = 128, 1, 4, 32, 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

def make_dataset(p):
    pairs = [(a, b) for a in range(p) for b in range(p)]
    inputs = torch.tensor([[a, b, p] for (a, b) in pairs])
    labels = torch.tensor([(a + b) % p for (a, b) in pairs])
    return inputs, labels

inputs, labels = make_dataset(P)
n_total = len(inputs)
perm = torch.randperm(n_total)
n_train = int(0.3 * n_total)  # small train fraction — this is what causes grokking
train_idx, test_idx = perm[:n_train], perm[n_train:]
train_inputs, train_labels = inputs[train_idx].to(DEVICE), labels[train_idx].to(DEVICE)
test_inputs, test_labels = inputs[test_idx].to(DEVICE), labels[test_idx].to(DEVICE)

cfg = HookedTransformerConfig(
    n_layers=N_LAYERS, d_model=D_MODEL, d_head=D_HEAD, n_heads=N_HEADS,
    d_mlp=4 * D_MODEL, d_vocab=P + 1, n_ctx=3, act_fn="relu",
    normalization_type="LN", seed=SEED,
)
model = HookedTransformer(cfg).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
# weight_decay=1.0 is unusually high on purpose — a known ingredient for reliable grokking

def loss_fn(logits, labels):
    return nn.functional.cross_entropy(logits[:, -1, :], labels)

def accuracy(logits, labels):
    return (logits[:, -1, :].argmax(dim=-1) == labels).float().mean().item()

N_STEPS, CHECKPOINT_EVERY = 15000, 100
checkpoints, log = {}, {"step": [], "train_acc": [], "test_acc": []}

for step in range(N_STEPS):
    model.train()
    optimizer.zero_grad()
    logits = model(train_inputs)
    loss = loss_fn(logits, train_labels)
    loss.backward()
    optimizer.step()

    if step % CHECKPOINT_EVERY == 0:
        model.eval()
        with torch.no_grad():
            train_acc = accuracy(logits, train_labels)
            test_acc = accuracy(model(test_inputs), test_labels)
        log["step"].append(step); log["train_acc"].append(train_acc); log["test_acc"].append(test_acc)
        checkpoints[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if step % 1000 == 0:
            print(f"Step {step}: train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

import pickle
pickle.dump(log, open("results/grokking_log.pkl", "wb"))
pickle.dump(checkpoints, open("results/grokking_checkpoints.pkl", "wb"))
```

**What to look for:** train accuracy hits ~100% early; test accuracy stays near 0% for a long stretch; then jumps to ~100% later, abruptly. That delayed jump is grokking.

**Adding LLC:** load each checkpoint back into the model and call `devinterp`'s `estimate_learning_coeff_with_summary`. Copy the exact API usage from the devinterp GitHub repo's own grokking example rather than guessing the function signature — their API can change between versions. Plot LLC alongside test accuracy; check whether LLC changes around the same step test accuracy jumps.

**Checkpoint:** done when you have one plot with train acc / test acc / LLC vs. step, and LLC shows something notable at the jump. If it doesn't line up, check devinterp's caveats docs and retry with their suggested defaults before concluding anything.

---

## Step 3: Reproduction Exercise B — Finding a Known Circuit (Activation Patching)

**Task:** reproduce the Indirect Object Identification (IOI) circuit in GPT-2 small (Wang et al. 2023, arXiv:2211.00593) — *"When Mary and John went to the store, John gave a drink to ___"* -> "Mary". The circuit is published, so you can check your work against ground truth.

```python
import torch
from transformer_lens import HookedTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2").to(DEVICE)
model.eval()

clean_prompt = "When Mary and John went to the store, John gave a drink to"
corrupt_prompt = "When John and Mary went to the store, John gave a drink to"
clean_tokens = model.to_tokens(clean_prompt)
corrupt_tokens = model.to_tokens(corrupt_prompt)
mary_token = model.to_single_token(" Mary")
john_token = model.to_single_token(" John")

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

def logit_diff(logits):
    final = logits[0, -1, :]
    return (final[mary_token] - final[john_token]).item()

print("Clean logit diff:", logit_diff(clean_logits))
print("Corrupt logit diff:", logit_diff(corrupt_logits))

n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
results = torch.zeros(n_layers, n_heads)

for layer in range(n_layers):
    for head in range(n_heads):
        def patch_head_hook(activation, hook, layer=layer, head=head):
            activation[:, -1, head, :] = clean_cache[f"blocks.{layer}.attn.hook_z"][:, -1, head, :]
            return activation
        patched_logits = model.run_with_hooks(
            corrupt_tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_z", patch_head_hook)])
        results[layer, head] = logit_diff(patched_logits)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.imshow(results.detach().cpu().numpy(), cmap="RdBu", aspect="auto")
plt.colorbar(label="Logit diff after patching")
plt.xlabel("Head"); plt.ylabel("Layer")
plt.title("Which heads, when patched, restore the 'Mary' answer?")
plt.savefig("results/ioi_patching_heatmap.png")
```

**Checkpoint:** done when your heatmap highlights a small number of heads, and at least some overlap with the Name Mover Heads / S-Inhibition Heads reported in Wang et al. Exact match isn't required — genuine overlap is your evidence the patching code works.

---

## Step 4: What NOT to do yet

Hold off on: building the causal-DAG task generator, RL post-training setup, ablation error-signature prediction, cross-topology transfer, and the implicit/explicit prompting gap. All come after this foundation is solid.

## Step 5: Week-3 Decision Point

1. Did grokking reproduce with a visible delayed jump?
2. Did LLC show something meaningful at that jump?
3. Did patching on IOI find heads overlapping the published circuit?

**Yes to all three ->** move on to the causal-DAG task generator.
**No to any ->** debug here first, or bring the specific failure to your mentor. A shaky foundation now silently invalidates everything later.

## Reference links
- TransformerLens docs: https://transformerlensorg.github.io/TransformerLens/
- devinterp GitHub (copy their grokking example exactly): https://github.com/timaeus-research/devinterp
- Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability" (2023)
- Wang et al., "Interpretability in the Wild: a Circuit for IOI in GPT-2 small" (arXiv:2211.00593)
