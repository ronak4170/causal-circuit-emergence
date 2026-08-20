"""
Decisive cheap diagnostic for Phase F, run BEFORE investing in any RFT-
preservation fix (rehearsal / KL-to-reference): does the RATIONALE warm-up
checkpoint -- BEFORE any RFT training, while it still reliably produces
rationale text (verified 20/20 in warmup_sft_rationale.py's own smoke
check) -- already clear the 0% floor on confounded-topology interventional
accuracy?

If NO: the failure isn't "RFT erodes the rationale before it can help" --
it's that the model can't compute the right answer even WITH the rationale
intact, and no amount of training-dynamics engineering (rehearsal, KL
penalty) will fix that. This settles the question with zero new training.

If YES: the rationale genuinely carries the missing capability, and it's
worth building a preservation mechanism (periodic rehearsal is the
recommended first choice -- see chat) so RFT doesn't train it away before
Phase F can benefit from it.

Reuses build_phase_c_test_set / extract_predicted_probability / is_correct
unchanged from prior phases (no new equations here, only a load-path
change: warmup_model_rationale.pt is FULL float32, saved via
torch.save(model.state_dict(), ...) with no .half() conversion, same as
every other warm-up checkpoint in this repo -- confirmed against
phase_a_main.py / phase_a_main_v2.py / phase_a_main_pythia.py, which all
load warm-up checkpoints the same direct way).
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


def load_model():
    model = HookedTransformer.from_pretrained(MODEL_NAME).to(DEVICE)
    state = torch.load(WARMUP_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
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
    print(f"Loading PRE-RFT rationale warm-up checkpoint from {WARMUP_CHECKPOINT}...")
    model = load_model()

    print("Building confounded-topology interventional test set (n=100)...")
    interv_test_set = build_phase_c_test_set(n_per_topology=100, seed=42, min_divergence=0.15)

    print("Evaluating interventional accuracy (decisive number)...")
    interv_results = evaluate(model, interv_test_set, "interventional_answer")
    n_correct = sum(1 for r in interv_results if is_correct(r["predicted"], r["true"], tolerance=0.10))
    interv_acc = n_correct / len(interv_results)
    print(f"\nPRE-RFT WARM-UP (RATIONALE) CONFOUNDED-TOPOLOGY INTERVENTIONAL ACCURACY: "
          f"{interv_acc:.3f} ({n_correct}/{len(interv_results)})")
    print("Compare to 0.000 for: original GPT-2 (post-RFT), Pythia-410M (post-RFT), "
          "Phase F rationale (post-RFT), Phase F control (post-RFT)")

    counts_per_topology = {"confounded": len(interv_test_set)}
    assoc_test_set = build_associational_test_set(counts_per_topology, seed=43)
    print("\nEvaluating associational accuracy (context)...")
    assoc_results = evaluate(model, assoc_test_set, "associational_answer")
    n_correct_assoc = sum(1 for r in assoc_results if is_correct(r["predicted"], r["true"], tolerance=0.10))
    assoc_acc = n_correct_assoc / len(assoc_results)
    print(f"ASSOCIATIONAL ACCURACY: {assoc_acc:.3f} ({n_correct_assoc}/{len(assoc_results)})")

    n_with_rationale = sum(1 for r in interv_results if len(r["generation"].strip()) > 25)
    print(f"\nGenerations longer than a bare answer (rationale-like content present): "
          f"{n_with_rationale}/{len(interv_results)}")

    print("\nSample generations (first 8 interventional):")
    for r in interv_results[:8]:
        print(f"  true={r['true']:.3f} pred={r['predicted']} | {r['generation']!r}")

    pickle.dump(
        {"interv_results": interv_results, "interv_acc": interv_acc,
         "assoc_results": assoc_results, "assoc_acc": assoc_acc,
         "n_with_rationale": n_with_rationale},
        open("results/phase_f_warmup_prerft_evaluation.pkl", "wb"),
    )
    print("\nSaved results/phase_f_warmup_prerft_evaluation.pkl")


if __name__ == "__main__":
    main()
