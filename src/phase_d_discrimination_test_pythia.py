"""
Pythia-410M version of phase_d_discrimination_test.py: does chain/fork/
collider's competence reflect genuine input-sensitive reasoning, or a
memorized near-constant that happens to fall within tolerance? Same test:
correlation between predicted and true answers, plus discrimination of
do_value=True vs False (does the mean prediction shift in the true
direction between the two groups?).
"""
import random

from causal_dag_task import sample_structural_equations, estimate_intervention, QA_CONFIG
from phase_b_setup_pythia import load_model
from phase_c_ablation_pythia import predicted_percentage_greedy
from rft_training_loop import format_prompt
from causal_dag_task import render_intervention_question

TOPOLOGIES_TO_TEST = ["chain", "fork", "collider"]
N_PER_TOPOLOGY = 40


def build_discrimination_test_set(n_per_topology=40, seed=99):
    rng = random.Random(seed)
    test_set = []
    for topo in TOPOLOGIES_TO_TEST:
        cfg = QA_CONFIG[topo]
        do_var, target_var = cfg["treat_var"], cfg["target_var"]
        for _ in range(n_per_topology):
            do_value = rng.choice([True, False])
            eqs, _ = sample_structural_equations(topo, rng)
            true_answer = estimate_intervention(eqs, target_var, do_var, do_value,
                                                 n_samples=4000, seed=rng.randint(0, 1_000_000))
            prompt = render_intervention_question(do_var, do_value, target_var)
            test_set.append({"topology": topo, "prompt": prompt, "true_answer": true_answer,
                              "do_value": do_value})
    return test_set


def pearson_corr(xs, ys):
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def main():
    print("Loading Pythia checkpoint 135 (final)...")
    model = load_model()

    print("Building discrimination test set...")
    test_set = build_discrimination_test_set(n_per_topology=N_PER_TOPOLOGY, seed=99)

    results = []
    for q in test_set:
        prompt = format_prompt(q["prompt"]) + " Answer:"
        pred = predicted_percentage_greedy(model, prompt)
        results.append({**q, "predicted": pred})

    print("\n--- Per-topology analysis ---\n")
    summary = {}
    for topo in TOPOLOGIES_TO_TEST:
        topo_results = [r for r in results if r["topology"] == topo and r["predicted"] is not None]
        n_valid = len(topo_results)
        preds = [r["predicted"] for r in topo_results]
        trues = [r["true_answer"] for r in topo_results]
        corr = pearson_corr(preds, trues)

        true_group = [r["predicted"] for r in topo_results if r["do_value"]]
        false_group = [r["predicted"] for r in topo_results if not r["do_value"]]
        true_true_group = [r["true_answer"] for r in topo_results if r["do_value"]]
        true_false_group = [r["true_answer"] for r in topo_results if not r["do_value"]]

        mean_pred_true = sum(true_group) / len(true_group) if true_group else None
        mean_pred_false = sum(false_group) / len(false_group) if false_group else None
        mean_true_true = sum(true_true_group) / len(true_true_group) if true_true_group else None
        mean_true_false = sum(true_false_group) / len(true_false_group) if true_false_group else None

        pred_diff = (mean_pred_true - mean_pred_false) if (mean_pred_true is not None and
                                                             mean_pred_false is not None) else None
        true_diff = (mean_true_true - mean_true_false) if (mean_true_true is not None and
                                                              mean_true_false is not None) else None
        direction_match = (pred_diff is not None and true_diff is not None and
                            (pred_diff > 0) == (true_diff > 0) and abs(pred_diff) > 0.02)

        print(f"[{topo}] n_valid={n_valid}/{N_PER_TOPOLOGY}")
        print(f"  correlation(predicted, true) = {corr}")
        print(f"  mean predicted | do_value=True:  {mean_pred_true}")
        print(f"  mean predicted | do_value=False: {mean_pred_false}")
        print(f"  model's pred_diff (True - False) = {pred_diff}")
        print(f"  true answer's true_diff (True - False) = {true_diff}")
        print(f"  direction match (model discriminates correctly): {direction_match}\n")

        summary[topo] = {
            "n_valid": n_valid, "correlation": corr,
            "mean_pred_true": mean_pred_true, "mean_pred_false": mean_pred_false,
            "mean_true_true": mean_true_true, "mean_true_false": mean_true_false,
            "pred_diff": pred_diff, "true_diff": true_diff,
            "direction_match": direction_match,
        }

    import pickle
    pickle.dump({"raw_results": results, "summary": summary},
                open("results/phase_d_pythia_discrimination_results.pkl", "wb"))
    print("Saved results/phase_d_pythia_discrimination_results.pkl")


if __name__ == "__main__":
    main()
