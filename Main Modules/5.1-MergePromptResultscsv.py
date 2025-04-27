#!/usr/bin/env python3
# merge_csvs.py
# Combine many CSVs that share the same columns into one master file.

import os
import csv
from pathlib import Path

# ---------- CHANGE THESE 2 LINES ONLY ----------
CSV_DIR        = Path("Main Modules/prompt_results")   # folder that holds the many CSVs
OUTPUT_CSV     = Path("merged_prompt_results.csv")     # where the merged file will be saved
# ------------------------------------------------

def collect_csv_paths(folder: Path) -> list[Path]:
    """Return every *.csv file inside `folder` (non-recursive)."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".csv")

def merge_csvs(csv_paths: list[Path], output_path: Path) -> None:
    if not csv_paths:
        print("❌ No CSV files found – nothing to merge.")
        return

    header_written = False
    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = None

        for csv_path in csv_paths:
            with open(csv_path, "r", newline="", encoding="utf-8") as fin:
                reader = csv.reader(fin)
                try:
                    header = next(reader)          # first row (column names)
                except StopIteration:
                    print(f"⚠️  {csv_path.name} is empty – skipping.")
                    continue

                # On first file, remember header & create writer
                if not header_written:
                    writer = csv.writer(fout)
                    writer.writerow(header)
                    header_written = True

                # For later files, verify header matches (optional)
                elif header != writer.writerows.__self__.fieldnames if hasattr(writer, 'writerows') else header:
                    # Basic check: same number of columns
                    if len(header) != len(writer.writerows.__self__.fieldnames if hasattr(writer, 'writerows') else header):
                        print(f"⚠️  Header mismatch in {csv_path.name} – skipping.")
                        continue  # or raise an error

                # Write the remaining lines (data rows)
                for row in reader:
                    writer.writerow(row)

            print(f"✓ merged {csv_path.name}")

    print(f"\n✅ All done. Combined file saved as {output_path}")

def main():
    csv_files = collect_csv_paths(CSV_DIR)
    merge_csvs(csv_files, OUTPUT_CSV)

if __name__ == "__main__":
    main()
