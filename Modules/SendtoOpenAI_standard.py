import os
import json
import time
from openai import OpenAI
import tiktoken

# Load API key
api_key = "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"
client = OpenAI(api_key=api_key)

# Config
INPUT_FILE = "prompts_short.jsonl"
OUTPUT_FILE = "batch_outputs/standard_output.jsonl"
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.7
RATE_LIMIT_DELAY = 1  # seconds
os.makedirs("batch_outputs", exist_ok=True)

# Token cost estimates (USD per 1K tokens)
TOKEN_COSTS = {
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002}
}

# Estimate tokens
def num_tokens_from_messages(messages, model):
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3  # each message includes metadata like {"role":..., "content":...}
    total_tokens = 0
    for message in messages:
        total_tokens += tokens_per_message
        for key, value in message.items():
            total_tokens += len(encoding.encode(value))
    total_tokens += 3  # assistant priming
    return total_tokens

def estimate_total_cost(prompts, model):
    total_input_tokens = 0
    estimated_output_tokens = 0  # You can tweak this if you expect longer replies

    for prompt in prompts:
        input_tokens = num_tokens_from_messages(prompt["body"]["messages"], model)
        total_input_tokens += input_tokens
        estimated_output_tokens += 200  # default guess per response

    token_cost = TOKEN_COSTS.get(model, TOKEN_COSTS["gpt-4"])
    input_cost = (total_input_tokens / 1000) * token_cost["prompt"]
    output_cost = (estimated_output_tokens / 1000) * token_cost["completion"]
    total_cost = input_cost + output_cost

    return total_input_tokens, estimated_output_tokens, total_cost

# Load prompts from file
def load_prompts(input_file):
    with open(input_file, "r") as f:
        return [json.loads(line) for line in f]

def send_prompt(client, prompt_obj):
    try:
        body = prompt_obj["body"]
        # Use chat.completions.create instead of completions.create
        response = client.chat.completions.create(
            model=body["model"],
            messages=body["messages"],
            temperature=body.get("temperature", 0.7)
        )
        return {
            "id": prompt_obj.get("custom_id", None),
            "prompt": prompt_obj,
            "response": response.choices[0].message.content  # Correct attribute access
        }
    except Exception as e:
        return {
            "id": prompt_obj.get("custom_id", None),
            "prompt": prompt_obj,
            "response": None,
            "error": str(e)
        }
# Save response to file
def save_response(output_file, response_obj):
    with open(output_file, "a") as f:
        f.write(json.dumps(response_obj) + "\n")

# Main pipeline
def main():
    prompts = load_prompts(INPUT_FILE)
    print(f"[INFO] Loaded {len(prompts)} prompts from {INPUT_FILE}")

    total_input_tokens, est_output_tokens, est_cost = estimate_total_cost(prompts, MODEL_NAME)

    print("\n=== Token & Cost Estimation ===")
    print(f"Estimated input tokens   : {total_input_tokens}")
    print(f"Estimated output tokens  : {est_output_tokens}")
    print(f"Estimated total cost (USD): ${est_cost:.4f}")
    print("==============================\n")

    confirm = input("Proceed with sending prompts? (yes/no): ").strip().lower()
    if confirm not in ["yes", "y"]:
        print("[INFO] Aborted by user.")
        return

    for idx, prompt_obj in enumerate(prompts):
        print(f"[INFO] Sending prompt {idx + 1}/{len(prompts)}...")
        result = send_prompt(client, prompt_obj)
        save_response(OUTPUT_FILE, result)
        time.sleep(RATE_LIMIT_DELAY)

    print(f"[INFO] All responses saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
