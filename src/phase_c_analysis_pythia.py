"""
Pythia-410M version of phase_c_analysis.py: same classification/statistical
tests (Predictions 1-3), pointed at the Pythia raw-results pickle.
"""
import pickle

from scipy.stats import binomtest

TIE_TOLERANCE = 0.02
ACCURACY_TOLERANCE = 0.10


def classify_ablated_outputs(results):
    classified = {"closer_to_interventional": 0, "closer_to_associational": 0,
                  "tie_or_unparseable": 0}
    per_topology = {}

    for r in results:
        pred = r["predicted"]
        if pred is None:
            classified["tie_or_unparseable"] += 1
            continue
        d_interv = abs(pred - r["interventional_true"])
        d_assoc = abs(pred - r["associational_true"])

        if abs(d_interv - d_assoc) < TIE_TOLERANCE:
            key = "tie_or_unparseable"
        elif d_assoc < d_interv:
            key = "closer_to_associational"
        else:
            key = "closer_to_interventional"

        classified[key] += 1
        per_topology.setdefault(r["topology"], {"assoc": 0, "interv": 0, "tie": 0})
        if key == "closer_to_associational":
            per_topology[r["topology"]]["assoc"] += 1
        elif key == "closer_to_interventional":
            per_topology[r["topology"]]["interv"] += 1
        else:
            per_topology[r["topology"]]["tie"] += 1

    return classified, per_topology


def is_correct(pred, true, tolerance=ACCURACY_TOLERANCE):
    if pred is None:
        return False
    return abs(pred - true) <= tolerance


def accuracy(results, true_key):
    n_correct = sum(1 for r in results if is_correct(r["predicted"], r[true_key]))
    return n_correct / len(results)


def main():
    d = pickle.load(open("results/phase_c_pythia_raw_results.pkl", "rb"))

    print("=" * 60)
    print("PREDICTION 1: Directional bias")
    print("=" * 60)
    classified, per_topology = classify_ablated_outputs(d["interv_ablated"])
    print(f"Classification: {classified}")

    n_assoc = classified["closer_to_associational"]
    n_interv = classified["closer_to_interventional"]
    n_decidable = n_assoc + n_interv
    frac_assoc = n_assoc / n_decidable if n_decidable > 0 else None

    binom_result = binomtest(n_assoc, n_decidable, p=0.5, alternative="greater")
    print(f"\nDecidable cases: {n_decidable} (ties/unparseable excluded: "
          f"{classified['tie_or_unparseable']})")
    print(f"Fraction closer to associational: {frac_assoc:.3f}")
    print(f"Binomial test p-value (H0: p=0.5, alt='greater'): {binom_result.pvalue:.6f}")

    p1_significant = binom_result.pvalue < 0.05
    p1_effect_size_met = frac_assoc is not None and frac_assoc >= 0.65
    p1_supported = p1_significant and p1_effect_size_met
    print(f"\nP1 statistical significance (p<0.05): {p1_significant}")
    print(f"P1 effect size threshold (>=0.65): {p1_effect_size_met}")
    print(f"P1 VERDICT: {'SUPPORTED' if p1_supported else 'NOT SUPPORTED'}")

    print("\n" + "=" * 60)
    print("PREDICTIONS 2 & 3: Selectivity and effect asymmetry")
    print("=" * 60)

    interv_acc_unablated = accuracy(d["interv_unablated"], "interventional_true")
    interv_acc_ablated = accuracy(d["interv_ablated"], "interventional_true")
    assoc_acc_unablated = accuracy(d["assoc_unablated"], "associational_true")
    assoc_acc_ablated = accuracy(d["assoc_ablated"], "associational_true")

    interv_drop = interv_acc_unablated - interv_acc_ablated
    assoc_drop = assoc_acc_unablated - assoc_acc_ablated

    print(f"Interventional accuracy: un-ablated={interv_acc_unablated:.3f}, "
          f"ablated={interv_acc_ablated:.3f}, drop={interv_drop:.3f}")
    print(f"Associational accuracy: un-ablated={assoc_acc_unablated:.3f}, "
          f"ablated={assoc_acc_ablated:.3f}, drop={assoc_drop:.3f}")

    p2_supported = abs(assoc_acc_ablated - assoc_acc_unablated) <= 0.10
    print(f"\nP2 (associational accuracy within 10pp of un-ablated): {p2_supported}")

    if assoc_drop > 0:
        asymmetry_ratio = interv_drop / assoc_drop
        p3_supported = asymmetry_ratio >= 2.0
        print(f"P3 asymmetry ratio (interv_drop / assoc_drop): {asymmetry_ratio:.3f}")
    else:
        asymmetry_ratio = float("inf") if interv_drop > 0 else None
        p3_supported = interv_drop > 0
        print(f"P3: associational drop <= 0 ({assoc_drop:.3f}); "
              f"asymmetry effectively {'infinite' if interv_drop > 0 else 'undefined'}")
    print(f"P3 VERDICT: {'SUPPORTED' if p3_supported else 'NOT SUPPORTED'}")

    pickle.dump({
        "classified": classified, "per_topology": per_topology,
        "frac_assoc": frac_assoc, "binom_pvalue": binom_result.pvalue,
        "p1_supported": p1_supported,
        "interv_acc_unablated": interv_acc_unablated, "interv_acc_ablated": interv_acc_ablated,
        "assoc_acc_unablated": assoc_acc_unablated, "assoc_acc_ablated": assoc_acc_ablated,
        "interv_drop": interv_drop, "assoc_drop": assoc_drop,
        "asymmetry_ratio": asymmetry_ratio,
        "p2_supported": p2_supported, "p3_supported": p3_supported,
    }, open("results/phase_c_pythia_analysis.pkl", "wb"))
    print("\nSaved results/phase_c_pythia_analysis.pkl")


if __name__ == "__main__":
    main()
