# section2_batch_submit.py
import openai
import time
import os

# Load your OpenAI API key from environment or set it here
openai.api_key = os.getenv("OPENAI_API_KEY")  # or hardcode like: openai.api_key = "sk-..."

# Files
INPUT_FILE = "prompts_full.jsonl"
BATCH_OUTPUT_DIR = "batch_outputs/"
os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)

# Submit the batch job
def submit_batch(input_file):
    with open(input_file, "rb") as f:
        batch = openai.Batch.create(
            input_file=f,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
    print(f"[INFO] Batch job submitted with ID: {batch.id}")
    return batch.id

# Check status until complete
def wait_for_completion(batch_id, poll_interval=20):
    print(f"[INFO] Waiting for batch {batch_id} to complete...")
    while True:
        batch = openai.Batch.retrieve(batch_id)
        print(f"  - Status: {batch.status}")
        if batch.status in ["completed", "failed", "expired", "cancelled"]:
            break
        time.sleep(poll_interval)
    return batch

# Download result
def download_results(batch):
    result_url = batch.output_file["url"]
    filename = f"{BATCH_OUTPUT_DIR}/batch_output_{batch.id}.jsonl"

    response = openai._client._client.request("GET", result_url)
    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"[INFO] Results saved to {filename}")
    return filename

# Run the full pipeline
def main():
    batch_id = submit_batch(INPUT_FILE)
    batch = wait_for_completion(batch_id)
    
    if batch.status == "completed":
        download_results(batch)
    else:
        print(f"[ERROR] Batch did not complete successfully. Status: {batch.status}")

if __name__ == "__main__":
    main()
