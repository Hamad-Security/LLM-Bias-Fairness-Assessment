#!/usr/bin/env python3
# upload_batches.py
# Upload several JSONL prompt files in a single run.

import os, re, json
from openai import OpenAI
import tiktoken

# ========== CONFIGURATION ========== # 3,4 ,12,13,5,14
JSONL_DIR     = "Main Modules/prompts"   # folder that holds your files
START_SEQ     = 1                        # first N to include
END_SEQ       = 1                       # last  N to include  (inclusive) 

MODEL         = "GPT-4.1-nano"
API_KEY       = "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"
BATCH_ID_FILE = "batch_ids_1.txt"          # one ID per line

# Pricing (per 1M tokens) and discount
PRICING = {
    "gpt-4o-mini":  {"input": 1.10, "output": 4.40},
    "GPT-4.1-nano": {"input": 0.10, "output": 0.400},
}
BATCH_DISCOUNT = 0.50           # 50 % off in batch
EST_OUT_TOKENS_PER_PROMPT = 100
# ===================================

# ---------- helper functions (unchanged) ----------
def get_token_counts(path: str) -> tuple[int, int]:
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
    cost_in  = in_toks / 1_000_000 * p["input"]  * BATCH_DISCOUNT
    cost_out = out_toks / 1_000_000 * p["output"] * BATCH_DISCOUNT
    return cost_in + cost_out


# create a single OpenAI client for the whole run
client = OpenAI(api_key=API_KEY)


def upload_batch(jsonl_path: str, model: str) -> str:
    """Upload one file and create one batch job, returning its ID."""
    # 1) upload the file
    with open(jsonl_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    # 2) create the batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch_job.id
# --------------------------------------------------


def collect_files() -> list[tuple[int, str]]:
    """
    Return a list [(seq, full_path), …] for every prompts_recent_<seq>.jsonl
    whose seq lies inside the configured range.
    """
    pat = re.compile(r"prompts_top-rated_(\d+)\.jsonl$")
    results: list[tuple[int, str]] = []
    for name in os.listdir(JSONL_DIR):
        m = pat.match(name)
        if not m:
            continue
        seq = int(m.group(1))
        if START_SEQ <= seq <= END_SEQ:
            results.append((seq, os.path.join(JSONL_DIR, name)))
    return sorted(results, key=lambda t: t[0])


def main() -> None:
    files = collect_files()
    if not files:
        print(f"❌ No files with sequence {START_SEQ}-{END_SEQ} found in {JSONL_DIR}")
        return

    # ---------- cost breakdown ----------
    tot_in = tot_out = 0
    print("\nFile             Input-tok   Output-tok  Est-Cost")
    print("-" * 50)
    for _, path in files:
        in_t, out_t = get_token_counts(path)
        tot_in  += in_t
        tot_out += out_t
        print(f"{os.path.basename(path):<20}{in_t:>10,}{out_t:>12,}")
    total_cost = estimate_cost(tot_in, tot_out, MODEL)
    print("-" * 50)
    print(f"TOTAL            {tot_in:>10,}{tot_out:>12,}   ≈ ${total_cost:,.4f}\n")
    # ------------------------------------

    if input("Proceed to upload ALL files? (y/n): ").lower() != "y":
        print("Aborted.")
        return

    # ---------- upload ----------
    with open(BATCH_ID_FILE, "a", encoding="utf-8") as f_ids:
        for seq, path in files:
            print(f"\nUploading {os.path.basename(path)} …")
            batch_id = upload_batch(path, MODEL)
            print(f"✅ seq {seq}: {batch_id}")
            f_ids.write(batch_id + "\n")
    # ----------------------------

    print(f"\nAll batch IDs appended to {BATCH_ID_FILE}")


if __name__ == "__main__":
    main()
