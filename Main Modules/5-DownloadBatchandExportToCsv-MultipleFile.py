#!/usr/bin/env python3
# download_batches.py
# Check status of many batch jobs, download results, convert to CSV.

import os, json, csv
from openai import OpenAI

# ========== CONFIGURATION ==========
API_KEY       = "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"
BATCH_ID_FILE = "batch_ids_1.txt"   # now contains many lines
RESULTS_DIR   = "prompt_results/top_rated"
# ===================================

client = OpenAI(api_key=API_KEY)

# ---------- helper functions (copied unchanged) ----------
def retrieve_status(batch_id: str):
    """Fetch full batch object for inspection."""
    batch = client.batches.retrieve(batch_id)
    print("🔍 Full batch response:\n", batch)
    return batch


def download_results(batch, save_dir: str = RESULTS_DIR) -> str:
    """
    Download the completed batch's JSONL output (or error file) by streaming
    the HttpxBinaryResponseContent into raw bytes.
    """
    file_id = getattr(batch, "output_file_id", None) or getattr(batch, "error_file_id", None)
    if not file_id:
        raise RuntimeError("❌ No output_file_id or error_file_id found in batch response.")

    response = client.files.content(file_id)
    chunks = []
    for chunk in response.iter_bytes(chunk_size=1024):
        if chunk:
            chunks.append(chunk)
    data = b"".join(chunks)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{batch.id}_results.jsonl")
    with open(path, "wb") as f:
        f.write(data)

    return path


def convert_jsonl_to_csv(jsonl_path: str, csv_path: str):
    """
    Convert one JSONL file to CSV, extracting 'custom_id' and 'movies' list.
    """
    with open(jsonl_path, "r", encoding="utf-8") as jsonl_file, \
         open(csv_path,  "w", newline="", encoding="utf-8") as csv_file:

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
                    movies_data = json.loads(content)
                    movies = movies_data.get("movies", [])
                    writer.writerow([custom_id, ", ".join(movies)])
                else:
                    writer.writerow([custom_id, ""])
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
# ---------------------------------------------------------


def main() -> None:
    if not os.path.exists(BATCH_ID_FILE):
        print(f"❌ Could not find {BATCH_ID_FILE}")
        return

    with open(BATCH_ID_FILE, encoding="utf-8") as f:
        batch_ids = [ln.strip() for ln in f if ln.strip()]

    if not batch_ids:
        print("File is empty – nothing to do.")
        return

    for idx, batch_id in enumerate(batch_ids, 1):
        print(f"\n=== [{idx}/{len(batch_ids)}] Batch {batch_id} ===")
        batch = retrieve_status(batch_id)
        status = getattr(batch, "status", "unknown")
        print(f"Status: {status}")

        if status != "completed":
            print("⏳ Not completed – skipping.")
            continue

        if input("Download results for this batch? (y/n): ").lower() != "y":
            print("Skipped download.")
            continue

        try:
            jsonl_path = download_results(batch)
            print(f"📥 Saved results to: {jsonl_path}")
            csv_path = os.path.splitext(jsonl_path)[0] + ".csv"
            convert_jsonl_to_csv(jsonl_path, csv_path)
            print(f"✅ Converted JSONL → CSV: {csv_path}")
        except Exception as e:
            print("⚠️ Error downloading or converting:", e)


if __name__ == "__main__":
    main()
