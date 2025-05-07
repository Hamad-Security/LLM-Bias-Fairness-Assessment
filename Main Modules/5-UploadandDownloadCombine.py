#!/usr/bin/env python3
"""
merged_batches.py (non‑interactive)
-----------------------------------
Upload JSONL prompt files to the OpenAI batch endpoint **two at a time**, wait
until *both* jobs in the pair finish, download their results, and convert them
to CSV before proceeding with the next pair. The sequence numbers that are
considered live are in the inclusive range [8, 25].

This variant is **completely non‑interactive**: it never pauses for user input.
If a pair of files is eligible it will be uploaded automatically, results will
be downloaded automatically when ready, and the script will then advance to the
next pair.
"""

from __future__ import annotations

from datetime import datetime
import os
import re
import json
import csv
import time
from typing import List, Tuple

from openai import OpenAI
import tiktoken

# ========== CONFIGURATION ========== 3 4 5 failed
JSONL_DIR     = "Main Modules/prompts"   # folder that holds your files
START_SEQ     = 15                         # first N to include (inclusive)
END_SEQ       = 25                        # last  N to include (inclusive)
PAIR_SIZE     = 2                         # how many files to upload at once

MODEL         = "GPT-4.1-nano"
API_KEY       = "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"

BATCH_ID_FILE = "batch_ids.txt"      # appends every created batch ID
RESULTS_DIR   = "prompt_results/top_rated"      # where *.jsonl and *.csv land

# Pricing (per 1 M tokens) and discount (unchanged)
PRICING = {
    "gpt-4o-mini":  {"input": 1.10, "output": 4.40},
    "GPT-4.1-nano": {"input": 0.10, "output": 0.400},
}
BATCH_DISCOUNT = 0.50           # 50 % off in batch
EST_OUT_TOKENS_PER_PROMPT = 100
POLL_INTERVAL_SEC = 60         # how often to ask for status (seconds)
# ===================================

client = OpenAI(api_key=API_KEY)

# ---------- helper functions ----------

def get_token_counts(path: str) -> tuple[int, int]:
    """Return (input_tokens, output_tokens ≈ constant) for one JSONL file."""
    enc = tiktoken.get_encoding("cl100k_base")
    in_toks = 0
    line_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line_count += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompt = obj["body"]["messages"][0]["content"]
                in_toks += len(enc.encode(prompt))
            except Exception as e:
                print(f"⚠️  Skipping line {ln} ({path}): {e}")
    out_toks = EST_OUT_TOKENS_PER_PROMPT * line_count
    return in_toks, out_toks


def estimate_cost(in_toks: int, out_toks: int, model: str) -> float:
    p = PRICING[model]
    return (
        in_toks / 1_000_000 * p["input"] * BATCH_DISCOUNT +
        out_toks / 1_000_000 * p["output"] * BATCH_DISCOUNT
    )


def upload_batch(jsonl_path: str, model: str) -> str:
    """Upload one file and create a batch job. Return its batch‑ID."""
    print(f"🚀 Uploading {os.path.basename(jsonl_path)} …")
    with open(jsonl_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"✅  Created batch {batch_job.id}")
    return batch_job.id


def retrieve_status(batch_id: str):
    return client.batches.retrieve(batch_id)


def download_results(batch, save_dir: str = RESULTS_DIR) -> str:
    file_id = getattr(batch, "output_file_id", None) or getattr(batch, "error_file_id", None)
    if not file_id:
        raise RuntimeError("❌  No output_file_id or error_file_id present in batch response.")

    response = client.files.content(file_id)
    data = b"".join(chunk for chunk in response.iter_bytes(chunk_size=1024) if chunk)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{batch.id}_results.jsonl")
    with open(path, "wb") as f:
        f.write(data)
    return path


def convert_jsonl_to_csv(jsonl_path: str) -> str:
    csv_path = os.path.splitext(jsonl_path)[0] + ".csv"
    with open(jsonl_path, "r", encoding="utf-8") as jsonl_file, \
         open(csv_path, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(["custom_id", "movies"])

        for line in jsonl_file:
            try:
                record = json.loads(line)
                custom_id = record.get("custom_id", "")
                body = record.get("response", {}).get("body", {})
                choices = body.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    movies = json.loads(content).get("movies", [])
                    writer.writerow([custom_id, ", ".join(movies)])
                else:
                    writer.writerow([custom_id, ""])
            except Exception as e:
                print(f"⚠️  Error processing line in {jsonl_path}: {e}")
    return csv_path


# --------------------------------------------------------


def collect_files() -> List[Tuple[int, str]]:
    pat = re.compile(r"prompts_top-rated_(\d+)\.jsonl$")
    matches: List[Tuple[int, str]] = []
    for name in os.listdir(JSONL_DIR):
        m = pat.match(name)
        if not m:
            continue
        seq = int(m.group(1))
        if START_SEQ <= seq <= END_SEQ:
            matches.append((seq, os.path.join(JSONL_DIR, name)))
    return sorted(matches, key=lambda t: t[0])


def pairwise(lst: List[Tuple[int, str]], size: int = PAIR_SIZE):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# =======================  main driver  =====================

def main() -> None:
    files = collect_files()
    if not files:
        print(f"❌  No files with sequence {START_SEQ}-{END_SEQ} found in {JSONL_DIR}")
        return

    print(f"Found {len(files)} matching files. Processing in groups of {PAIR_SIZE}…\n")

    with open(BATCH_ID_FILE, "a", encoding="utf-8") as f_ids:
        for pair in pairwise(files, PAIR_SIZE):
            # —— cost breakdown for this pair ——
            tot_in = tot_out = 0
            print("Files in current pair      Input-tok   Output-tok   Est-Cost")
            print("-" * 60)
            for _, path in pair:
                in_t, out_t = get_token_counts(path)
                tot_in  += in_t
                tot_out += out_t
                print(f"{os.path.basename(path):<25}{in_t:>10,}{out_t:>12,}")
            cost = estimate_cost(tot_in, tot_out, MODEL)
            print("-" * 60)
            print(f"TOTAL{tot_in:>29,}{tot_out:>12,}   ≈ ${cost:,.4f}\n")
            # —— automatically continue ——

            batch_ids: List[str] = [upload_batch(path, MODEL) for _, path in pair]
            for bid in batch_ids:
                f_ids.write(bid + "\n")

            remaining = set(batch_ids)
            while remaining:
                time.sleep(POLL_INTERVAL_SEC)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time+" ⏳  Polling status…")
                for bid in list(remaining):
                    batch = retrieve_status(bid)
                    status = getattr(batch, "status", "unknown")
                    print(f"Batch {bid}: {status}")
                    if status == "completed":
                        try:
                            jl_path = download_results(batch)
                            print(f"📥  Results saved → {jl_path}")
                            csv_path = convert_jsonl_to_csv(jl_path)
                            print(f"📑  Converted to CSV → {csv_path}")
                        except Exception as e:
                            print(f"⚠️  Error downloading/converting for {bid}:", e)
                        remaining.discard(bid)
                    elif status in {"failed", "expired"}:
                        print(f"❌  Batch {bid} ended with status: {status}. Moving on.")
                        remaining.discard(bid)
                print()

            print("🎉  Pair finished. Moving to next…\n")

    print("✅  All eligible files have been processed.")


if __name__ == "__main__":
    main()
