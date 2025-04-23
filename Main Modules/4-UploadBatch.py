import os
import json
from openai import OpenAI
import tiktoken

# ========== CONFIGURATION ==========
JSONL_FILE    = "Main Modules/prompts/prompts_random.jsonl"  # your JSONL file
MODEL         = "gpt-4o-mini"                              # or "gpt-4.1-mini", etc.
API_KEY       = "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"

BATCH_ID_FILE = "batch_id.txt"

# Pricing (per 1M tokens) and discount
PRICING = {
    "gpt-4o-mini": {"input": 1.10, "output": 4.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}
BATCH_DISCOUNT = 0.50  # 50% off in batch
EST_OUT_TOKENS_PER_PROMPT = 100
# ====================================

# Initialize client
client = OpenAI(api_key=API_KEY)

def get_token_counts(path: str) -> tuple[int,int]:
    enc = tiktoken.get_encoding("cl100k_base")
    in_toks = out_toks = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompt = obj["body"]["messages"][0]["content"]
                in_toks += len(enc.encode(prompt))
                out_toks += EST_OUT_TOKENS_PER_PROMPT
            except Exception as e:
                print(f"Skipping line {ln}: {e}")
    return in_toks, out_toks

def estimate_cost(in_toks: int, out_toks: int, model: str) -> float:
    p = PRICING[model]
    cost_in  = in_toks  / 1_000_000 * p["input"]  * BATCH_DISCOUNT
    cost_out = out_toks  / 1_000_000 * p["output"] * BATCH_DISCOUNT
    return cost_in + cost_out

def upload_batch(jsonl_path: str, model: str) -> str:
    # 1) upload the file
    with open(jsonl_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    # 2) create the batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return batch_job.id

def main():
    if not os.path.exists(JSONL_FILE):
        print(f"❌ File not found: {JSONL_FILE}")
        return

    in_toks, out_toks = get_token_counts(JSONL_FILE)
    cost = estimate_cost(in_toks, out_toks, MODEL)
    print(f"Tokens → input: {in_toks:,}, output: {out_toks:,} | est. cost: ${cost:.4f} USD")

    if input("Proceed to upload? (y/n): ").lower() != "y":
        print("Aborted.")
        return

    print("Uploading batch…")
    batch_id = upload_batch(JSONL_FILE, MODEL)
    print(f"✅ Batch created: {batch_id}")

    with open(BATCH_ID_FILE, "w") as f:
        f.write(batch_id)
    print(f"Batch ID saved to {BATCH_ID_FILE}")

if __name__ == "__main__":
    main()
