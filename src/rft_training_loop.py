"""
Rejection-sampling fine-tuning (RFT) loop for the causal-DAG task.
This is the RL post-training regime for Phase A.
"""
import re

import torch
import torch.nn as nn


def format_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer (as a percentage, e.g. 'Answer: 42%'):"


def extract_predicted_probability(generated_text: str):
    """Parse a percentage out of the model's generation. Return None if unparseable."""
    match = re.search(r"(\d{1,3})\s*%", generated_text)
    if match:
        val = int(match.group(1))
        if 0 <= val <= 100:
            return val / 100.0
    return None


def is_correct(predicted_prob, oracle_prob, tolerance=0.10):
    if predicted_prob is None:
        return False
    return abs(predicted_prob - oracle_prob) <= tolerance


def rft_iteration(model, task_batch, device, tolerance=0.10,
                   n_samples_per_question=4, temperature=0.8, verbose=False):
    """
    task_batch: list of dicts with keys {"question": str, "oracle_answer": float, "rung": str}
    Returns: list of (prompt, accepted_generation_text) pairs for fine-tuning,
             plus per-rung accuracy stats for logging.
    """
    accepted_pairs = []
    stats = {"association": [0, 0], "intervention": [0, 0], "counterfactual": [0, 0]}

    for item in task_batch:
        prompt = format_prompt(item["question"])
        rung = item["rung"]
        stats[rung][1] += 1

        found_correct = False
        for _ in range(n_samples_per_question):
            tokens = model.to_tokens(prompt)
            generated = model.generate(tokens, max_new_tokens=15,
                                        temperature=temperature, do_sample=True,
                                        verbose=False)
            # Slice on token count, not string length: to_string(generated[0])[len(prompt):]
            # is WRONG because detokenization isn't length-invariant (e.g. the prepended
            # BOS token shifts character offsets), so a naive string slice can include the
            # tail of the prompt itself -- and since every prompt contains the literal
            # example text "Answer: 42%", the parser would silently latch onto that instead
            # of the model's real answer.
            new_tokens = generated[0, tokens.shape[1]:]
            gen_text = model.to_string(new_tokens)
            pred = extract_predicted_probability(gen_text)

            if is_correct(pred, item["oracle_answer"], tolerance):
                accepted_pairs.append((prompt, gen_text))
                if not found_correct:
                    stats[rung][0] += 1
                    found_correct = True
                break

    return accepted_pairs, stats


def finetune_on_accepted(model, optimizer, accepted_pairs, device):
    """Standard cross-entropy fine-tuning on the accepted (prompt, generation) pairs."""
    if len(accepted_pairs) == 0:
        return None

    total_loss = 0.0
    model.train()
    for prompt, generation in accepted_pairs:
        full_text = prompt + generation
        tokens = model.to_tokens(full_text).to(device)
        logits = model(tokens)
        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            tokens[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    model.eval()

    return total_loss / len(accepted_pairs)
