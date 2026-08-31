import json
import os
from datasets import load_dataset
import random

def main():
    # Set seed for reproducibility
    random.seed(42)
    
    print("Loading datasets...")
    # Load GSM8K (harder math problems)
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    # Take 75 samples
    gsm8k_samples = list(gsm8k)
    random.shuffle(gsm8k_samples)
    gsm8k_samples = gsm8k_samples[:75]
    
    # Load MMLU (easier/mixed knowledge questions, we'll pick a few subjects)
    # Using a subset for simplicity
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    mmlu_samples = list(mmlu)
    random.shuffle(mmlu_samples)
    mmlu_samples = mmlu_samples[:75]
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "eval_set.jsonl")
    
    print(f"Preparing {len(gsm8k_samples) + len(mmlu_samples)} total prompts...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Process GSM8K
        for i, row in enumerate(gsm8k_samples):
            record = {
                "id": f"gsm8k_{i}",
                "prompt": row["question"],
                "ground_truth": row["answer"].split("####")[-1].strip(), # Extract the final answer
                "source": "gsm8k"
            }
            f.write(json.dumps(record) + "\n")
            
        # Process MMLU
        for i, row in enumerate(mmlu_samples):
            choices = row["choices"]
            answer_idx = row["answer"]
            
            # Format as multiple choice
            prompt = f"{row['question']}\n\nChoices:\n"
            for j, choice in enumerate(choices):
                prompt += f"{chr(65+j)}. {choice}\n"
            prompt += "\nPlease respond with just the letter (A, B, C, or D) of the correct answer."
            
            record = {
                "id": f"mmlu_{i}",
                "prompt": prompt,
                "ground_truth": chr(65+answer_idx), # E.g., 'A', 'B', 'C', 'D'
                "source": "mmlu"
            }
            f.write(json.dumps(record) + "\n")
            
    print(f"Successfully wrote data to {output_path}")

if __name__ == "__main__":
    main()
