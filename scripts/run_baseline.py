import json
import os
import csv
import time
import requests
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def evaluate_answer(raw_answer, ground_truth, source):
    """
    Very basic evaluation logic.
    For MMLU, we expect A, B, C, or D. We check if the ground truth is in the output (ignoring case usually, but it should be exact).
    For GSM8K, we check if the exact number string appears in the output.
    """
    if not raw_answer:
        return False
        
    # Simple check: Does the ground truth string exist in the raw answer?
    return str(ground_truth).strip().lower() in raw_answer.strip().lower()

def main():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in environment.")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Configuration
    model_name = os.getenv("CAPABLE_MODEL", "gemini-3.1-flash-lite")
    
    # Paths
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, "data", "eval_set.jsonl")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "baseline.csv")
    
    print(f"Reading dataset from {data_path}")
    prompts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
                
    # Run on a small sample first if requested, but for now we'll just process what's in the list
    # The instructions say "Test small before scaling: run the full 5-stage pipeline on 5 prompts first"
    # We will slice here for testing if needed, or process all. Let's start with 5 for safety, 
    # but the script should process all. We'll use a TEST_MODE flag.
    
    TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"
    if TEST_MODE:
        print("TEST_MODE enabled. Running on 5 prompts only.")
        prompts = prompts[:5]
        
    print(f"Running baseline evaluation on {len(prompts)} prompts...")
    
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "model", "input_tokens", "output_tokens", "latency_ms", "raw_answer", "correct"])
        
        for item in prompts:
            prompt_id = item["id"]
            user_prompt = item["prompt"]
            ground_truth = item["ground_truth"]
            source = item["source"]
            
            print(f"Processing {prompt_id}...")
            
            # Rate limiting / Retry logic
            max_retries = 3
            retry_delay = 5 # base delay
            
            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    
                    response = client.models.generate_content(
                        model=model_name,
                        contents=user_prompt,
                    )
                    
                    end_time = time.time()
                    latency_ms = int((end_time - start_time) * 1000)
                    
                    raw_answer = response.text
                    
                    # Token counting (using rough estimate or API provided metrics if available)
                    # Gemini GenAI SDK usage metadata
                    input_tokens = 0
                    output_tokens = 0
                    if response.usage_metadata:
                        input_tokens = response.usage_metadata.prompt_token_count
                        output_tokens = response.usage_metadata.candidates_token_count
                    
                    correct = evaluate_answer(raw_answer, ground_truth, source)
                    
                    writer.writerow([prompt_id, model_name, input_tokens, output_tokens, latency_ms, raw_answer.replace("\n", " "), correct])
                    f.flush()
                    break # Success, exit retry loop
                    
                except Exception as e:
                    print(f"Error on {prompt_id} (Attempt {attempt+1}/{max_retries}): {e}")
                    if "429" in str(e) or "quota" in str(e).lower():
                        if attempt < max_retries - 1:
                            sleep_time = retry_delay * (2 ** attempt)
                            print(f"Rate limited. Retrying in {sleep_time}s...")
                            time.sleep(sleep_time)
                        else:
                            print(f"Failed {prompt_id} after {max_retries} attempts.")
                            writer.writerow([prompt_id, model_name, 0, 0, 0, f"ERROR: {str(e)}", False])
                            f.flush()
                    else:
                        writer.writerow([prompt_id, model_name, 0, 0, 0, f"ERROR: {str(e)}", False])
                        f.flush()
                        break # Not a rate limit error, don't retry immediately

    print(f"Baseline run complete. Results saved to {results_path}")

if __name__ == "__main__":
    main()
