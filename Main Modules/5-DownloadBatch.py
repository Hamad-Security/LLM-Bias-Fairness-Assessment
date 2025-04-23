import os
from openai import OpenAI

# ========== CONFIGURATION ==========
API_KEY       = "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"
BATCH_ID_FILE = "batch_id.txt"
RESULTS_DIR   = "prompt_results"
# ====================================

# Initialize client
client = OpenAI(api_key=API_KEY)

def retrieve_status(batch_id: str):
    """Fetch full batch object for inspection."""
    batch = client.batches.retrieve(batch_id)
    print("🔍 Full batch response:\n", batch)
    return batch

def download_results(batch, save_dir: str = RESULTS_DIR) -> str:
    """
    Download the completed batch's JSONL output (or error file) by
    streaming the HttpxBinaryResponseContent into raw bytes.
    """
    client = OpenAI(api_key=API_KEY)

    # Prefer the successful output file, else fall back to the error file
    file_id = getattr(batch, "output_file_id", None) or getattr(batch, "error_file_id", None)
    if not file_id:
        raise RuntimeError("❌ No output_file_id or error_file_id found in batch response.")

    # 1) Fetch the streaming response object
    response = client.files.content(file_id)
    # 2) Collect raw bytes in chunks
    chunks = []
    for chunk in response.iter_bytes(chunk_size=1024):
        if chunk:
            chunks.append(chunk)
    # 3) Concatenate into a single bytes object
    data = b"".join(chunks)

    # 4) Write to disk
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{batch.id}_results.jsonl")
    with open(path, "wb") as f:
        f.write(data)

    return path
def main():
    if not os.path.exists(BATCH_ID_FILE):
        print(f"❌ Could not find {BATCH_ID_FILE}")
        return

    with open(BATCH_ID_FILE) as f:
        batch_id = f.read().strip()

    batch = retrieve_status(batch_id)
    status = getattr(batch, "status", "unknown")
    print(f"\n📦 Batch Status: {status}")

    if status == "completed":
        if input("Download results? (y/n): ").lower() == "y":
            try:
                path = download_results(batch)
                print(f"📥 Saved results to: {path}")
            except Exception as e:
                print("⚠️ Error downloading:", e)
        else:
            print("Skipped download.")
    else:
        print("⏳ Not completed yet. Retry later.")

if __name__ == "__main__":
    main()
