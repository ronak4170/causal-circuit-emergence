"""
Build matched implicit/explicit intervention prompt pairs for Phase D.

Deviation from the starter code, disclosed (same lesson learned the hard way
in Phase C): the starter's topology_var_map offers multiple (do_var,
target_var) choices per topology, some of which are NOT the pair the model
was actually trained on in Phase A (causal_dag_task.QA_CONFIG). Testing
prompts built from an untrained pair would confound "does the model lack
interventional competence" with "the model has never seen this exact
variable combination phrased this way" -- an out-of-distribution problem,
not a scaffolding problem. So this version uses ONLY each topology's actual
trained (treat_var, target_var) pair from QA_CONFIG.
"""
import random

from causal_dag_task import sample_structural_equations, estimate_intervention, QA_CONFIG

IMPLICIT_TEMPLATE = ("Suppose we directly force {var_desc} to be {val}, regardless of what "
                     "would normally determine it. What is the probability that {target_desc} "
                     "is on?")

EXPLICIT_TEMPLATES = [
    ("This is a question about intervention, not observation. When we intervene to force "
     "{var_desc} to be {val}, we should ignore any factors that would normally cause "
     "{var_desc}. Given the intervention, what is the probability that {target_desc} is on?"),
    ("The following is an intervention question. To answer it correctly, treat {var_desc} as "
     "if it is being externally set by us, meaning we should ignore anything that would "
     "normally influence {var_desc}. Given that {var_desc} is now {val}, what is the "
     "probability that {target_desc} is on?"),
    ("Question type: intervention (do-operator). Notation: we are computing "
     "P({target_desc} | do({var_desc} = {val})). This differs from P({target_desc} | "
     "{var_desc} = {val}) because we should ignore the normal causes of {var_desc}. "
     "What is the probability?"),
]

VAR_NAMES = {"A": "switch A", "B": "switch B", "C": "light C", "U": "hidden factor U"}


def build_prompt_pair(topology, do_var, do_value, target_var, rng):
    eqs, _ = sample_structural_equations(topology, rng)
    interv_answer = estimate_intervention(eqs, target_var, do_var, do_value,
                                           n_samples=4000, seed=rng.randint(0, 1_000_000))

    var_desc = VAR_NAMES[do_var]
    target_desc = VAR_NAMES[target_var]
    val = "on" if do_value else "off"

    implicit = IMPLICIT_TEMPLATE.format(var_desc=var_desc, val=val, target_desc=target_desc)
    explicits = [t.format(var_desc=var_desc, val=val, target_desc=target_desc)
                 for t in EXPLICIT_TEMPLATES]

    return {
        "topology": topology, "implicit": implicit, "explicit_variants": explicits,
        "true_answer": interv_answer,
        "meta": {"do_var": do_var, "do_value": do_value, "target_var": target_var},
    }


def build_prompt_pair_set(n_per_topology=15, seed=1234):
    rng = random.Random(seed)
    pairs = []
    for topo, cfg in QA_CONFIG.items():
        do_var, target_var = cfg["treat_var"], cfg["target_var"]
        for _ in range(n_per_topology):
            do_value = rng.choice([True, False])
            pairs.append(build_prompt_pair(topo, do_var, do_value, target_var, rng))
    return pairs


if __name__ == "__main__":
    pairs = build_prompt_pair_set(n_per_topology=15, seed=1234)
    print(f"Total pairs: {len(pairs)}")

    print("\n--- Smell-test: 6 random pairs (Step 2.4 checkpoint) ---\n")
    rng = random.Random(0)
    sample = rng.sample(pairs, 6)
    for p in sample:
        print(f"[{p['topology']}] true_answer={p['true_answer']:.3f}")
        print(f"  IMPLICIT: {p['implicit']}")
        for i, e in enumerate(p["explicit_variants"]):
            print(f"  EXPLICIT v{i+1}: {e}")
        print()

    lengths_implicit = [len(p["implicit"]) for p in pairs]
    lengths_explicit = [len(e) for p in pairs for e in p["explicit_variants"]]
    print(f"Implicit prompt length (chars): mean={sum(lengths_implicit)/len(lengths_implicit):.0f}")
    print(f"Explicit prompt length (chars): mean={sum(lengths_explicit)/len(lengths_explicit):.0f}")
