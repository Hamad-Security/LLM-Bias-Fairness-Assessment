'''# ---------------------------------------------------------------
# Quick-stats extractor for the Movie-ratings dataset
# ---------------------------------------------------------------
import pandas as pd
from pathlib import Path

########################################################################
# 1.  CONFIG – point to your real file
########################################################################
CSV_PATH = Path("Dataset/full_movies_data.csv")          # <-- update to the real name/location

########################################################################
# 2.  LOAD & TIDY
########################################################################
df = pd.read_csv(CSV_PATH)

# Convert UNIX seconds → pandas datetime for easy date maths
df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)

########################################################################
# 3.  CORE METRICS (the ones used in the canvas doc)
########################################################################
metrics = {
    "rows"          : len(df),
    "unique_users"  : df["UserID"].nunique(),
    "unique_movies" : df["MovieID"].nunique(),
    "time_span"     : f"{df['Datetime'].min().date()} → {df['Datetime'].max().date()}",
    "rating_mean"   : round(df["Rating"].mean(), 2),
    "rating_std"    : round(df["Rating"].std(ddof=1), 2),
    "duplicates"    : int(df.duplicated().sum()),
    # show only columns with ≥1 missing values
    "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n}
}

########################################################################
# 4.  DISPLAY NICELY
########################################################################
print("\n📊  DATASET OVERVIEW")
for k, v in metrics.items():
    print(f"{k:15}: {v}")

########################################################################
# 5.  (OPTIONAL) RETURN AS DICT IF YOU’D RATHER USE IT PROGRAMMATICALLY
########################################################################
def dataset_overview(frame: pd.DataFrame) -> dict:
    """Return the same dict of headline stats for any MovieLens-style DataFrame."""
    out = metrics.copy()
    return out

# Example usage:
# stats = dataset_overview(df)

import pandas as pd
import re

# ---------- 1 · Load -------------------------------------------------
PATH = "Dataset/full_movies_data.csv"          # ← change if your file lives elsewhere
df   = pd.read_csv(PATH)

# ---------- 2 · Extract release year from the Title -----------------
# Titles look like "Forrest Gump (1994)" – grab the 4-digit part
df["Year"] = (
    df["Title"]
    .str.extract(r"\((\d{4})\)", expand=False)  # returns string
    .astype(int)
)

# ---------- 3 · Locate min / max years ------------------------------
min_year, max_year = df["Year"].min(), df["Year"].max()

# ---------- 4 · Pull the movie lists --------------------------------
movies_min_year = df.loc[df["Year"] == min_year, "Title"].unique()
movies_max_year = df.loc[df["Year"] == max_year, "Title"].unique()

# ---------- 5 · Display results -------------------------------------
print(f"📅 Earliest year in dataset  : {min_year}")
print("   Movies released that year:")
for title in movies_min_year:
    print("   •", title)

print(f"\n📅 Latest year in dataset    : {max_year}")
print("   Movies released that year:")
for title in movies_max_year:
    print("   •", title)
'''

# ---------------------------------------------------------------
# Histogram of movie counts (density) · Years 1919 – 2000
# ---------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1 · LOAD  ------------------------------------------------------
CSV_PATH = "Dataset/full_movies_data.csv"          # ← change if your file lives elsewhere
df = pd.read_csv(CSV_PATH)

# 2 · EXTRACT YEAR  ---------------------------------------------
df["Year"] = (
    df["Title"]
      .str.extract(r"\((\d{4})\)", expand=False)   # pull the 4-digit year string
      .astype(int)
)

# 3 · FILTER RANGE  ---------------------------------------------
year_mask = (df["Year"] >= 1919) & (df["Year"] <= 2000)
years = df.loc[year_mask, "Year"]

# 4 · PLOT  ------------------------------------------------------
# Histogram (probability-density)
plt.figure(figsize=(8,4))
counts, bins, patches = plt.hist(
    years,
    bins=range(1919, 2001),       # one bin per calendar year
    density=True,                 # y-axis shows density, not raw counts
    edgecolor="black",
)

# Optional: smoothed kernel-density curve ------------------------
try:
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(years, bw_method="scott")
    x = np.linspace(years.min(), years.max(), 500)
    plt.plot(x, kde(x), linewidth=2)
except ImportError:
    # scipy not installed – skip the line and just keep the histogram
    pass

# 5 · LABELS & AESTHETICS  --------------------------------------
plt.title("Distribution of Movie Release Years (1919 – 2000)")
plt.xlabel("Release year")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
