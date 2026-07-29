"""
Follow-up to Phase D: isolate the disclosed length confound (explicit
prompts averaged ~1.8x longer than implicit, 245 vs 137 chars).

Adds a third condition per pair: a "neutral_control" prompt -- the same
implicit question, padded with neutral, content-free filler text to match
that pair's own explicit-variant length. This lets us separate:
  - explicit vs neutral_control: isolates SCAFFOLDING CONTENT (same length,
    only one has causal-reasoning hints)
  - neutral_control vs implicit: isolates LENGTH alone (same content, only
    one is padded)
If explicit beats neutral_control by about as much as it beat implicit
before, the original apparent effects were about content. If neutral_control
performs as well as explicit, the original effects were just about length.
"""
import random

from phase_d_prompt_pairs import build_prompt_pair_set

NEUTRAL_FILLER = ("Please consider this question carefully and think about it for a moment "
                   "before answering. Take your time to review the details. ")


def add_neutral_control(pair):
    target_len = sum(len(e) for e in pair["explicit_variants"]) / len(pair["explicit_variants"])
    base = pair["implicit"]
    padding_needed = max(0, int(target_len) - len(base))
    filler = (NEUTRAL_FILLER * (padding_needed // len(NEUTRAL_FILLER) + 1))[:padding_needed]
    pair["neutral_control"] = base + " " + filler
    return pair


def build_prompt_pair_set_with_control(n_per_topology=15, seed=1234):
    pairs = build_prompt_pair_set(n_per_topology=n_per_topology, seed=seed)
    for p in pairs:
        add_neutral_control(p)
    return pairs


if __name__ == "__main__":
    pairs = build_prompt_pair_set_with_control(n_per_topology=15, seed=1234)
    rng = random.Random(0)
    sample = rng.sample(pairs, 3)
    for p in sample:
        print(f"[{p['topology']}] true_answer={p['true_answer']:.3f}")
        print(f"  IMPLICIT ({len(p['implicit'])} chars): {p['implicit']}")
        print(f"  NEUTRAL_CONTROL ({len(p['neutral_control'])} chars): {p['neutral_control']}")
        print(f"  EXPLICIT v1 ({len(p['explicit_variants'][0])} chars): {p['explicit_variants'][0]}")
        print()

    lengths_implicit = [len(p["implicit"]) for p in pairs]
    lengths_neutral = [len(p["neutral_control"]) for p in pairs]
    lengths_explicit = [len(e) for p in pairs for e in p["explicit_variants"]]
    print(f"Mean lengths -- implicit: {sum(lengths_implicit)/len(lengths_implicit):.0f}, "
          f"neutral_control: {sum(lengths_neutral)/len(lengths_neutral):.0f}, "
          f"explicit: {sum(lengths_explicit)/len(lengths_explicit):.0f}")
