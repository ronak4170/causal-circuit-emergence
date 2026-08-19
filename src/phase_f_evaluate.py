"""
Phase F behavioral evaluation: does either warm-up condition (rationale vs.
control) clear the 0% floor on confounded-topology interventional accuracy
that both the original GPT-2 and Pythia-410M runs hit (see
results/phase_c_conclusion.md, results/phase_c_pythia_conclusion.md)?

Uses free-form generation (not the single-forward-pass greedy method Phase
B/C/D use), since a rationale-trained model's actual answer no longer sits
at a fixed token position right after "...Answer:" -- it can appear anywhere
after a variable-length rationale. extract_predicted_probability searches
the whole generation for a percentage, so this works for both conditions
even though only one produces rationale text in practice.

Run with: python phase_f_evaluate.py [rationale|control] [checkpoint_iteration]
Defaults to the final checkpoint (iteration 135, matching Phase A's convention).
"""
import pickle
import sys

import torch
from transformer_lens import HookedTransformer

from phase_c_reference_answers import build_phase_c_test_set, build_associational_test_set
from rft_training_loop import format_prompt, extract_predicted_probability, is_correct

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"
MAX_NEW_TOKENS = 60


def load_model(condition, checkpoint_iter=135):
    checkpoint_dir = f"results/phase_f_{condition}_checkpoints"
    checkpoint_path = f"{checkpoint_dir}/ckpt_{checkpoint_iter:04d}.pt"
    model = HookedTransformer.from_pretrained(MODEL_NAME).to(DEVICE)
    half_state = torch.load(checkpoint_path, map_location=DEVICE)
    full_state = {k: v.float() for k, v in half_state.items()}
    model.load_state_dict(full_state)
    model.eval()
    return model


def predicted_percentage_generate(model, prompt):
    tokens = model.to_tokens(prompt)
    generated = model.generate(tokens, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0,
                                do_sample=False, verbose=False)
    new_tokens = generated[0, tokens.shape[1]:]
    gen_text = model.to_string(new_tokens)
    return extract_predicted_probability(gen_text), gen_text


def evaluate(model, test_questions, true_key):
    results = []
    for q in test_questions:
        prompt = format_prompt(q["prompt"]) + " Answer:"
        pred, gen_text = predicted_percentage_generate(model, prompt)
        results.append({"topology": q["topology"], "predicted": pred,
                         "true": q[true_key], "generation": gen_text})
    return results


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("rationale", "control"):
        print("Usage: python phase_f_evaluate.py [rationale|control] [checkpoint_iteration]")
        sys.exit(1)
    condition = sys.argv[1]
    checkpoint_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 135

    print(f"Loading {condition} checkpoint (iteration {checkpoint_iter})...")
    model = load_model(condition, checkpoint_iter)

    print("Building confounded-topology interventional test set (n=100)...")
    interv_test_set = build_phase_c_test_set(n_per_topology=100, seed=42, min_divergence=0.15)

    print("Evaluating interventional accuracy (headline number)...")
    interv_results = evaluate(model, interv_test_set, "interventional_answer")
    n_correct = sum(1 for r in interv_results if is_correct(r["predicted"], r["true"], tolerance=0.10))
    interv_acc = n_correct / len(interv_results)
    print(f"\nCONFOUNDED-TOPOLOGY INTERVENTIONAL ACCURACY: {interv_acc:.3f} "
          f"({n_correct}/{len(interv_results)}) -- compare to 0.000 for both the original "
          f"GPT-2 and Pythia-410M runs")

    counts_per_topology = {"confounded": len(interv_test_set)}
    assoc_test_set = build_associational_test_set(counts_per_topology, seed=43)
    print("\nEvaluating associational accuracy (selectivity check)...")
    assoc_results = evaluate(model, assoc_test_set, "associational_answer")
    n_correct_assoc = sum(1 for r in assoc_results if is_correct(r["predicted"], r["true"], tolerance=0.10))
    assoc_acc = n_correct_assoc / len(assoc_results)
    print(f"ASSOCIATIONAL ACCURACY: {assoc_acc:.3f} ({n_correct_assoc}/{len(assoc_results)})")

    print("\nSample generations (first 5 interventional):")
    for r in interv_results[:5]:
        print(f"  true={r['true']:.3f} pred={r['predicted']} | {r['generation']!r}")

    pickle.dump(
        {"condition": condition, "checkpoint_iter": checkpoint_iter,
         "interv_results": interv_results, "interv_acc": interv_acc,
         "assoc_results": assoc_results, "assoc_acc": assoc_acc},
        open(f"results/phase_f_{condition}_evaluation.pkl", "wb"),
    )
    print(f"\nSaved results/phase_f_{condition}_evaluation.pkl")


if __name__ == "__main__":
    main()
