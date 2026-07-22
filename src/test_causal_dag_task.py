"""Step 1.5 checkpoint: verify the task generator produces genuine, meaningful
association/intervention divergence where the design requires it, before any
training code is written. Per instructions, do not proceed to Step 2 until
this passes.
"""
import random

from causal_dag_task import (
    sample_structural_equations, estimate_association, estimate_intervention,
    check_collider_explaining_away, generate_instance, QA_CONFIG,
)

TOPOLOGIES = ["chain", "fork", "collider", "confounded"]


def test_generate_20_each():
    rng = random.Random(0)
    for topo in TOPOLOGIES:
        for rung in ["association", "intervention", "counterfactual"]:
            for _ in range(20):
                inst = generate_instance(topo, rung, rng)
                assert inst["oracle_answer"] is not None, f"{topo}/{rung} produced no answer"
                assert 0.0 <= inst["oracle_answer"] <= 1.0
    print("[PASS] Generated 20 instances of each topology x rung with valid oracle answers.")


def test_confounded_divergence():
    """REQUIRED checkpoint: for 'confounded', P(B|A) must differ meaningfully
    from P(B|do(A)) -- this is the only structure where the effect of A on B
    is genuinely confounded by U."""
    rng = random.Random(1)
    diffs = []
    for trial in range(10):
        eqs, _ = sample_structural_equations("confounded", rng)
        p_assoc = estimate_association(eqs, "B", "A", True, n_samples=20000, seed=trial * 2)
        p_interv = estimate_intervention(eqs, "B", "A", True, n_samples=20000, seed=trial * 2 + 1)
        diffs.append(abs(p_assoc - p_interv))
        print(f"  trial {trial}: P(B=T|A=T)={p_assoc:.3f}  P(B=T|do(A=T))={p_interv:.3f}  "
              f"diff={abs(p_assoc - p_interv):.3f}")
    mean_diff = sum(diffs) / len(diffs)
    assert mean_diff > 0.10, (
        f"Confounding too weak: mean |P(B|A) - P(B|do(A))| = {mean_diff:.3f} <= 0.10. "
        "Increase confounder strength (lower noise on U->A / U->B)."
    )
    print(f"[PASS] confounded topology: mean association/intervention divergence = {mean_diff:.3f}")


def test_chain_and_fork_controls_agree():
    """Sanity check on the OTHER side of the design: for chain (do_var=B on a
    real B->C edge) and fork (do_var=A on a real root->child edge), there is
    no confounding of the chosen (treat_var, target_var) pair, so association
    and intervention should closely agree -- these are the deliberate
    'no divergence' control cases."""
    rng = random.Random(2)
    for topo in ["chain", "fork"]:
        cfg = QA_CONFIG[topo]
        diffs = []
        for trial in range(10):
            eqs, _ = sample_structural_equations(topo, rng)
            p_assoc = estimate_association(eqs, cfg["target_var"], cfg["treat_var"], True,
                                             n_samples=20000, seed=trial * 2)
            p_interv = estimate_intervention(eqs, cfg["target_var"], cfg["treat_var"], True,
                                              n_samples=20000, seed=trial * 2 + 1)
            diffs.append(abs(p_assoc - p_interv))
        mean_diff = sum(diffs) / len(diffs)
        assert mean_diff < 0.08, f"{topo} control case diverged unexpectedly: {mean_diff:.3f}"
        print(f"[PASS] {topo} control case: mean divergence = {mean_diff:.3f} (small, as expected)")


def test_collider_explaining_away():
    """REQUIRED checkpoint: conditioning on the collider C should induce
    association between A and B where none exists unconditionally."""
    rng = random.Random(3)
    diffs = []
    for trial in range(10):
        eqs, _ = sample_structural_equations("collider", rng)
        p_marg, p_given_c_true, p_given_c_false = check_collider_explaining_away(
            eqs, n_samples=20000, seed=trial * 3)
        diff = abs(p_given_c_true - p_given_c_false)
        diffs.append(diff)
        print(f"  trial {trial}: P(B)={p_marg:.3f}  P(B|C=T)={p_given_c_true:.3f}  "
              f"P(B|C=F)={p_given_c_false:.3f}  |gap|={diff:.3f}")
    mean_diff = sum(diffs) / len(diffs)
    assert mean_diff > 0.05, (
        f"Explaining-away effect too weak: mean gap = {mean_diff:.3f} <= 0.05."
    )
    print(f"[PASS] collider explaining-away: mean |P(B|C=T)-P(B|C=F)| = {mean_diff:.3f}")


if __name__ == "__main__":
    test_generate_20_each()
    print()
    test_confounded_divergence()
    print()
    test_chain_and_fork_controls_agree()
    print()
    test_collider_explaining_away()
    print("\nAll Step 1.5 checkpoint tests passed.")
