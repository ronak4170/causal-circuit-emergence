"""
Decisive cheap diagnostic for Phase F, run BEFORE investing in any RFT-
preservation fix (rehearsal / KL-to-reference): does the RATIONALE warm-up
checkpoint -- BEFORE any RFT training -- ever produce a CORRECT answer on
confounded-topology interventional questions when it actually writes the
rationale?

v2 fix (v1 was flawed): v1 used greedy decoding (temperature=0.0), which
turned out to suppress the rationale entirely -- the model's single highest-
probability continuation was a bare answer + degenerate repeat loop, even
though warmup_sft_rationale.py's own smoke check (temperature=0.8,
do_sample=True, matching how RFT itself samples) showed the rationale
appears reliably under stochastic decoding. v1's "n_with_rationale" metric
was also broken -- it used generation LENGTH as a proxy, but the degenerate
repeat loop is also long, so it falsely counted as "rationale present."

v2 samples N_SAMPLES completions per question at TEMPERATURE=0.8 (matching
phase_f_main.py's own RFT sampling settings exactly, for a fair test of
"could RFT ever have found and reinforced a correct rationale-containing
completion"), and detects rationale presence via a real content match: the
fixed opening phrase "This is an intervention" is a near-verbatim substring
from phase_f_rationale_templates.build_intervention_rationale's template
(deterministic, not model-paraphrased), so its presence in a generation is
a reliable signal, not a heuristic.

Reports THREE numbers:
  1. Best-of-N accuracy overall (any of the N samples correct) -- the
     decisive number: could RFT-style sampling ever have found a correct
     completion to reinforce, regardless of whether it contained a rationale?
  2. Accuracy specifically among generations that DID contain the rationale.
  3. Accuracy specifically among generations that did NOT contain it.
If (2) is meaningfully above 0 while (3) stays near 0, the rationale is
doing real causal work and is worth preserving through RFT. If both stay
at/near 0, the rationale carries no answer-correctness benefit regardless
of RFT, and no preservation mechanism will help.

Reuses build_phase_c_test_set / extract_predicted_probability / is_correct
unchanged from prior phases; load-path is full float32 direct load, same
as phase_a_main.py / phase_a_main_v2.py / phase_a_main_pythia.py.
"""
import pickle

import torch
from transformer_lens import HookedTransformer

from phase_c_reference_answers import build_phase_c_test_set, build_associational_test_set
from rft_training_loop import format_prompt, extract_predicted_probability, is_correct

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"
MAX_NEW_TOKENS = 60
WARMUP_CHECKPOINT = "results/warmup_model_rationale.pt"
N_SAMPLES = 4          # matches phase_f_main.py's N_SAMPLES_PER_QUESTION
TEMPERATURE = 0.8      # matches phase_f_main.py's TEMPERATURE
RATIONALE_MARKER = "This is an intervention"


def load_model():
    model = HookedTransformer.from_pretrained(MODEL_NAME).to(DEVICE)
    state = torch.load(WARMUP_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def sample_completions(model, prompt, n_samples=N_SAMPLES, temperature=TEMPERATURE):
    tokens = model.to_tokens(prompt)
    completions = []
    for _ in range(n_samples):
        generated = model.generate(tokens, max_new_tokens=MAX_NEW_TOKENS, temperature=temperature,
                                    do_sample=True, verbose=False)
        new_tokens = generated[0, tokens.shape[1]:]
        gen_text = model.to_string(new_tokens)
        pred = extract_predicted_probability(gen_text)
        has_rationale = RATIONALE_MARKER in gen_text
        completions.append({"predicted": pred, "generation": gen_text, "has_rationale": has_rationale})
    return completions


def evaluate_bestof_n(model, test_questions, true_key, tolerance=0.10):
    results = []
    for q in test_questions:
        prompt = format_prompt(q["prompt"]) + " Answer:"
        true_answer = q[true_key]
        completions = sample_completions(model, prompt)
        for c in completions:
            c["correct"] = is_correct(c["predicted"], true_answer, tolerance=tolerance)
        any_correct = any(c["correct"] for c in completions)
        results.append({"topology": q["topology"], "true": true_answer,
                         "completions": completions, "any_correct": any_correct})
    return results


def main():
    print(f"Loading PRE-RFT rationale warm-up checkpoint from {WARMUP_CHECKPOINT}...")
    model = load_model()

    print("Building confounded-topology interventional test set (n=100)...")
    interv_test_set = build_phase_c_test_set(n_per_topology=100, seed=42, min_divergence=0.15)

    print(f"Evaluating interventional accuracy: {N_SAMPLES} samples/question at "
          f"temperature={TEMPERATURE} (matching phase_f_main.py's own RFT sampling)...")
    interv_results = evaluate_bestof_n(model, interv_test_set, "interventional_answer")

    n_best_of_n_correct = sum(1 for r in interv_results if r["any_correct"])
    best_of_n_acc = n_best_of_n_correct / len(interv_results)

    all_completions = [c for r in interv_results for c in r["completions"]]
    with_rationale = [c for c in all_completions if c["has_rationale"]]
    without_rationale = [c for c in all_completions if not c["has_rationale"]]
    acc_with_rationale = (sum(1 for c in with_rationale if c["correct"]) / len(with_rationale)
                           if with_rationale else None)
    acc_without_rationale = (sum(1 for c in without_rationale if c["correct"]) / len(without_rationale)
                              if without_rationale else None)

    print(f"\nBEST-OF-{N_SAMPLES} CONFOUNDED-TOPOLOGY INTERVENTIONAL ACCURACY: "
          f"{best_of_n_acc:.3f} ({n_best_of_n_correct}/{len(interv_results)})")
    print(f"Total completions sampled: {len(all_completions)} "
          f"({len(with_rationale)} contained the rationale marker, "
          f"{len(without_rationale)} did not)")
    print(f"Per-completion accuracy WITH rationale present: "
          f"{acc_with_rationale if acc_with_rationale is not None else 'n/a (none produced)'}")
    print(f"Per-completion accuracy WITHOUT rationale present: "
          f"{acc_without_rationale if acc_without_rationale is not None else 'n/a (none produced)'}")
    print("Compare best-of-N accuracy to 0.000 for: original GPT-2 (post-RFT), "
          "Pythia-410M (post-RFT), Phase F rationale (post-RFT), Phase F control (post-RFT)")

    counts_per_topology = {"confounded": len(interv_test_set)}
    assoc_test_set = build_associational_test_set(counts_per_topology, seed=43)
    print("\nEvaluating associational accuracy (context, greedy single-sample)...")
    assoc_correct = 0
    assoc_results = []
    for q in assoc_test_set:
        prompt = format_prompt(q["prompt"]) + " Answer:"
        tokens = model.to_tokens(prompt)
        generated = model.generate(tokens, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0,
                                    do_sample=False, verbose=False)
        gen_text = model.to_string(generated[0, tokens.shape[1]:])
        pred = extract_predicted_probability(gen_text)
        correct = is_correct(pred, q["associational_answer"], tolerance=0.10)
        assoc_correct += correct
        assoc_results.append({"predicted": pred, "true": q["associational_answer"],
                               "generation": gen_text, "correct": correct})
    assoc_acc = assoc_correct / len(assoc_test_set)
    print(f"ASSOCIATIONAL ACCURACY (greedy): {assoc_acc:.3f} ({assoc_correct}/{len(assoc_test_set)})")

    print("\nSample completions (first 3 questions, all samples shown):")
    for r in interv_results[:3]:
        print(f"  true={r['true']:.3f}")
        for c in r["completions"]:
            print(f"    pred={c['predicted']} correct={c['correct']} rationale={c['has_rationale']} "
                  f"| {c['generation']!r}")

    pickle.dump(
        {"interv_results": interv_results, "best_of_n_acc": best_of_n_acc,
         "acc_with_rationale": acc_with_rationale, "acc_without_rationale": acc_without_rationale,
         "n_with_rationale": len(with_rationale), "n_without_rationale": len(without_rationale),
         "assoc_results": assoc_results, "assoc_acc": assoc_acc},
        open("results/phase_f_warmup_prerft_evaluation.pkl", "wb"),
    )
    print("\nSaved results/phase_f_warmup_prerft_evaluation.pkl")


if __name__ == "__main__":
    main()
