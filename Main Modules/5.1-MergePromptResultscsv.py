#!/usr/bin/env python3
# merge_csvs.py  – revised, simpler, no AttributeError

import csv
from pathlib import Path

# ------------ EDIT THESE 2 LINES ------------
CSV_DIR    = Path("prompt_results/top_rated")   # folder containing the many CSVs
OUTPUT_CSV = Path("LlmCsvOutput/merged_prompt_results_top_rated.csv")     # where to save the combined file
# --------------------------------------------

def collect_csv_paths(folder: Path) -> list[Path]:
    """Return every *.csv file inside `folder` (flat, not recursive)."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".csv")

def merge_csvs(csv_paths: list[Path], output_path: Path) -> None:
    if not csv_paths:
        print("❌ No CSV files found – nothing to merge.")
        return

    first_header: list[str] | None = None
    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)

        for csv_path in csv_paths:
            with open(csv_path, "r", newline="", encoding="utf-8") as fin:
                reader = csv.reader(fin)

                try:
                    header = next(reader)      # read the header row
                except StopIteration:
                    print(f"⚠️  {csv_path.name} is empty – skipping.")
                    continue

                if first_header is None:
                    # first file → remember header & write it once
                    first_header = header
                    writer.writerow(header)
                else:
                    # later files → make sure the header matches
                    if header != first_header:
                        print(f"⚠️  Header mismatch in {csv_path.name} – skipping.")
                        continue

                # copy the remaining rows
                for row in reader:
                    writer.writerow(row)

            print(f"✓ merged {csv_path.name}")

    print(f"\n✅ All done. Combined file saved as {output_path}")

def main():
    csv_files = collect_csv_paths(CSV_DIR)
    merge_csvs(csv_files, OUTPUT_CSV)

if __name__ == "__main__":
    main()
