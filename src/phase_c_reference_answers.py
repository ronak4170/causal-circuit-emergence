"""
Step 2: for each intervention test question, compute both the true
interventional answer and the true associational answer, so ablated-model
outputs can later be scored against both (per PREREGISTRATION.md's Phase C
predictions, committed in e2ae1c7 before this file was written).
"""
import random

from causal_dag_task import (sample_structural_equations, estimate_association,
                              estimate_intervention, render_intervention_question,
                              render_association_question)

# Restricted to confounded topology only, per the PREREGISTRATION.md addendum: the
# (treat_var, target_var) pairs the model was ACTUALLY TRAINED ON in Phase A
# (causal_dag_task.QA_CONFIG) show ~0 association/intervention divergence for
# chain/fork/collider by design (Step 1.3's deliberate unconfounded control cases).
# Only confounded's (A, B) pair has real divergence (~0.41), so it's the only
# topology where Prediction 1's directional test is even meaningful in-distribution.
TOPOLOGY_VAR_MAP = {
    "confounded": [("A", "B")],
}


def build_phase_c_test_set(n_per_topology=100, seed=42, min_divergence=0.15):
    rng = random.Random(seed)
    test_questions = []

    for topology, var_options in TOPOLOGY_VAR_MAP.items():
        collected = 0
        attempts = 0
        while collected < n_per_topology and attempts < n_per_topology * 20:
            attempts += 1
            do_var, target_var = rng.choice(var_options)
            do_value = rng.choice([True, False])
            eqs, _ = sample_structural_equations(topology, rng)

            assoc = estimate_association(eqs, target_var, do_var, do_value,
                                          n_samples=4000, seed=rng.randint(0, 1_000_000))
            interv = estimate_intervention(eqs, target_var, do_var, do_value,
                                            n_samples=4000, seed=rng.randint(0, 1_000_000))

            if assoc is None or interv is None:
                continue
            if abs(assoc - interv) < min_divergence:
                continue

            test_questions.append({
                "topology": topology,
                "prompt": render_intervention_question(do_var, do_value, target_var),
                "interventional_answer": interv,
                "associational_answer": assoc,
                "divergence": abs(assoc - interv),
                "do_var": do_var, "do_value": do_value, "target_var": target_var,
                "eqs": eqs,
            })
            collected += 1

        print(f"{topology}: collected {collected} usable questions from {attempts} attempts")

    return test_questions


def build_associational_test_set(counts_per_topology, seed=43):
    """Pure association-rung questions, count matched per topology to the
    interventional test set (for Predictions 2/3)."""
    rng = random.Random(seed)
    test_questions = []
    for topology, n in counts_per_topology.items():
        var_options = TOPOLOGY_VAR_MAP[topology]
        collected = 0
        attempts = 0
        while collected < n and attempts < n * 20:
            attempts += 1
            given_var, target_var = rng.choice(var_options)
            given_value = rng.choice([True, False])
            eqs, _ = sample_structural_equations(topology, rng)
            assoc = estimate_association(eqs, target_var, given_var, given_value,
                                          n_samples=4000, seed=rng.randint(0, 1_000_000))
            if assoc is None:
                continue
            test_questions.append({
                "topology": topology,
                "prompt": render_association_question(given_var, given_value, target_var),
                "associational_answer": assoc,
                "given_var": given_var, "given_value": given_value, "target_var": target_var,
                "eqs": eqs,
            })
            collected += 1
        print(f"{topology}: collected {collected} associational questions from {attempts} attempts")
    return test_questions


if __name__ == "__main__":
    test_set = build_phase_c_test_set(n_per_topology=100, seed=42, min_divergence=0.15)
    print(f"\nTotal interventional test questions: {len(test_set)}")

    divergences = [q["divergence"] for q in test_set]
    print(f"Divergence stats: min={min(divergences):.3f}, max={max(divergences):.3f}, "
          f"mean={sum(divergences)/len(divergences):.3f}")

    by_topo = {}
    for q in test_set:
        by_topo.setdefault(q["topology"], []).append(q["divergence"])
    print("\nPer-topology counts and mean divergence:")
    for topo, divs in by_topo.items():
        print(f"  {topo}: n={len(divs)}, mean divergence={sum(divs)/len(divs):.3f}")
