"""Builds three disjoint prompt sets so the calibrator is never evaluated on
data it trained on:

  - calibration_train.jsonl (100 prompts): fed to collect_calibration_data.py
    to fit the calibrator.
  - eval_holdout.jsonl (50 prompts): used only by run_eval.py, never seen during
    training - this is what proves calibration generalizes rather than memorizes.
  - hard_ood.jsonl (20 hand-written prompts): open-research-problem-style
    questions (proofs, specialized-knowledge, ambiguous judgment calls) that a
    naive router usually under-rates. Used both to build the "known-hard cluster"
    embedding centroid and as a stress test in run_eval.py's ablation.

Same GSM8K + MMLU source pull pattern as v1's scripts/prepare_dataset.py, just
sliced into non-overlapping chunks instead of one flat eval_set.jsonl.
"""
import json
import os
import random

from datasets import load_dataset

SEED = 42
N_TRAIN = 50  # per source (GSM8K + MMLU), so 100 total
N_EVAL = 25  # per source, so 50 total


HARD_OOD_PROMPTS = [
    "Prove that there are infinitely many prime numbers using Euclid's argument, showing every step.",
    "Prove Fermat's Last Theorem for n=4 using infinite descent, with full rigor.",
    "Write a formal proof that P != NP.",
    "Derive the Euler-Lagrange equation from the principle of stationary action for a functional depending on a function and its first derivative.",
    "Explain the exact quantum field theory calculation for the anomalous magnetic moment of the electron to 5 loops.",
    "Translate this sentence into classical Sanskrit using correct sandhi rules: The wise king protects his people.",
    "Prove that the square root of 2 is irrational using a formal proof by contradiction, showing every logical step.",
    "Derive the closed-form solution to a second-order linear homogeneous ODE with complex roots, including full derivation.",
    "Give a rigorous proof of the Cayley-Hamilton theorem for an arbitrary n x n matrix over a field.",
    "Explain in full technical detail how zero-knowledge succinct non-interactive arguments of knowledge (zk-SNARKs) achieve succinctness, including the polynomial commitment scheme.",
    "Derive the Navier-Stokes equations from first principles starting from conservation of momentum in a continuum.",
    "Prove the Banach fixed-point theorem and explain its role in proving existence/uniqueness for ODEs.",
    "Give a formal semantics for a simply-typed lambda calculus with a soundness proof for its type system.",
    "Explain the renormalization group flow of the coupling constant in phi^4 theory near the Wilson-Fisher fixed point.",
    "Derive the Chern-Gauss-Bonnet theorem for a compact Riemannian manifold without boundary.",
    "Prove Godel's first incompleteness theorem at the level of rigor expected in a graduate logic course.",
    "Explain the exact combinatorial proof of the hook length formula for the number of standard Young tableaux.",
    "Derive the Black-Scholes partial differential equation from the assumption of a geometric Brownian motion price process and a replicating portfolio argument.",
    "Give a rigorous treatment of Sylow's theorems and prove the first one in full.",
    "Explain the derivation of the Wess-Zumino-Witten term in the context of chiral perturbation theory.",
]


def main():
    random.seed(SEED)
    print("Loading GSM8K and MMLU...")
    gsm8k = list(load_dataset("openai/gsm8k", "main", split="test"))
    mmlu = list(load_dataset("cais/mmlu", "all", split="test"))
    random.shuffle(gsm8k)
    random.shuffle(mmlu)

    total_needed = N_TRAIN + N_EVAL
    gsm8k_slice = gsm8k[:total_needed]
    mmlu_slice = mmlu[:total_needed]

    def gsm8k_record(i, row):
        return {
            "id": f"gsm8k_{i}",
            "prompt": row["question"],
            "ground_truth": row["answer"].split("####")[-1].strip(),
            "source": "gsm8k",
        }

    def mmlu_record(i, row):
        choices = row["choices"]
        answer_idx = row["answer"]
        prompt = f"{row['question']}\n\nChoices:\n"
        for j, choice in enumerate(choices):
            prompt += f"{chr(65 + j)}. {choice}\n"
        prompt += "\nPlease respond with just the letter (A, B, C, or D) of the correct answer."
        return {
            "id": f"mmlu_{i}",
            "prompt": prompt,
            "ground_truth": chr(65 + answer_idx),
            "source": "mmlu",
        }

    train_records = [gsm8k_record(i, r) for i, r in enumerate(gsm8k_slice[:N_TRAIN])] + [
        mmlu_record(i, r) for i, r in enumerate(mmlu_slice[:N_TRAIN])
    ]
    eval_records = [
        gsm8k_record(i, r) for i, r in enumerate(gsm8k_slice[N_TRAIN:total_needed], start=N_TRAIN)
    ] + [mmlu_record(i, r) for i, r in enumerate(mmlu_slice[N_TRAIN:total_needed], start=N_TRAIN)]

    train_ids = {r["id"] for r in train_records}
    eval_ids = {r["id"] for r in eval_records}
    assert train_ids.isdisjoint(eval_ids), "train/eval split leaked overlapping ids"

    hard_ood_records = [
        {"id": f"hard_{i}", "prompt": p, "ground_truth": None, "source": "hard_ood"}
        for i, p in enumerate(HARD_OOD_PROMPTS)
    ]

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    for name, records in [
        ("calibration_train.jsonl", train_records),
        ("eval_holdout.jsonl", eval_records),
        ("hard_ood.jsonl", hard_ood_records),
    ]:
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(records)} records to {path}")


if __name__ == "__main__":
    main()
