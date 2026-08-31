import json
import os
import csv
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/route-demo")

def main():
    # Paths
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, "data", "eval_set.jsonl")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "routed.csv")
    
    print(f"Reading dataset from {data_path}")
    prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
                
    TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"
    if TEST_MODE:
        print("TEST_MODE enabled. Running on 5 prompts only.")
        prompts = prompts[:5]
        
    print(f"Running routed evaluation on {len(prompts)} prompts...")
    print(f"Targeting webhook: {N8N_WEBHOOK_URL}")
    
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "model", "difficulty_score", "input_tokens", "output_tokens", "classifier_input_tokens", "classifier_output_tokens", "latency_ms", "raw_answer", "correct"])
        
        for item in prompts:
            prompt_id = item["id"]
            
            print(f"Processing {prompt_id}...")
            
            payload = {
                "id": prompt_id,
                "prompt": item["prompt"],
                "ground_truth": item["ground_truth"],
                "source": item["source"]
            }
            
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    
                    response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=60)
                    response.raise_for_status()
                    
                    end_time = time.time()
                    latency_ms = int((end_time - start_time) * 1000)
                    
                    data = response.json()
                    
                    writer.writerow([
                        data.get("id", prompt_id),
                        data.get("model", "unknown"),
                        data.get("difficulty_score", 0),
                        data.get("input_tokens", 0),
                        data.get("output_tokens", 0),
                        data.get("classifier_input_tokens", 0),
                        data.get("classifier_output_tokens", 0),
                        latency_ms,
                        data.get("raw_answer", "").replace("\n", " "),
                        data.get("correct", False)
                    ])
                    f.flush()
                    break # Success
                    
                except requests.exceptions.RequestException as e:
                    print(f"Error on {prompt_id} (Attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        sleep_time = retry_delay * (2 ** attempt)
                        print(f"Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        print(f"Failed {prompt_id} after {max_retries} attempts.")
                        writer.writerow([prompt_id, "ERROR", 0, 0, 0, 0, 0, 0, f"ERROR: {str(e)}", False])
                        f.flush()

    print(f"Routed run complete. Results saved to {results_path}")

if __name__ == "__main__":
    main()
