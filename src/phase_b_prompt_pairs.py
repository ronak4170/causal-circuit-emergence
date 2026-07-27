"""
Step 2: build clean/corrupted prompt pairs for intervention questions across
all four DAG topologies, reusing the oracle from causal_dag_task.py.

Minimal pair design: same DAG, same target variable, flip the intervention's
forced value (True vs False). Only pairs where the oracle's clean/corrupt
answers differ by > 0.15 are kept -- otherwise patching has nothing to
detect.
"""
import random

from causal_dag_task import sample_structural_equations, estimate_intervention, render_intervention_question

TOPOLOGY_VAR_MAP = {
    "chain": [("A", "C"), ("B", "C")],
    "fork": [("A", "B"), ("A", "C")],
    "collider": [("A", "C"), ("B", "C")],
    # (A, B) is deliberately excluded here: under do(A), the U-confound is
    # severed, so forcing A True vs False has ~zero effect on B (both
    # converge to B's unconditional marginal) -- this is CORRECT deconfounding
    # behavior, not a bug, but it means that pair essentially never clears the
    # divergence filter and is useless for patching's minimal-pair need. (A,C)
    # is the pair with a genuine direct causal edge, so it's kept instead.
    "confounded": [("A", "C")],
}


def build_intervention_pair(topology, target_var, do_var, rng):
    eqs, edges = sample_structural_equations(topology, rng)

    clean_prompt = render_intervention_question(do_var, True, target_var)
    corrupt_prompt = render_intervention_question(do_var, False, target_var)

    clean_answer = estimate_intervention(eqs, target_var, do_var, True,
                                          n_samples=4000, seed=rng.randrange(1 << 30))
    corrupt_answer = estimate_intervention(eqs, target_var, do_var, False,
                                            n_samples=4000, seed=rng.randrange(1 << 30))

    return {
        "topology": topology, "clean_prompt": clean_prompt, "corrupt_prompt": corrupt_prompt,
        "clean_answer": clean_answer, "corrupt_answer": corrupt_answer,
        "eqs": eqs, "target_var": target_var, "do_var": do_var,
    }


def build_pair_batch(n_per_topology=15, seed=0, min_gap=0.15):
    rng = random.Random(seed)
    pairs = []
    counts = {topo: 0 for topo in TOPOLOGY_VAR_MAP}
    for topo, var_options in TOPOLOGY_VAR_MAP.items():
        for _ in range(n_per_topology):
            do_var, target_var = rng.choice(var_options)
            pair = build_intervention_pair(topo, target_var, do_var, rng)
            if abs(pair["clean_answer"] - pair["corrupt_answer"]) > min_gap:
                pairs.append(pair)
                counts[topo] += 1
    return pairs, counts


if __name__ == "__main__":
    pairs, counts = build_pair_batch(n_per_topology=15, seed=0)
    print(f"Total usable pairs: {len(pairs)} / {15 * len(TOPOLOGY_VAR_MAP)} generated")
    for topo, n in counts.items():
        print(f"  {topo}: {n}/15 survived the >0.15 divergence filter")
