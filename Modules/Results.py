# section3_evaluate_fairness.py
import json
import pandas as pd
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# Constants
OUTPUT_FILE = "batch_outputs/batch_output_<your_batch_id>.jsonl"  # Replace with actual ID
USER_METADATA_FILE = "user_metadata.csv"  # Contains user_id, gender, age_group, etc.

# Load batch results
def load_results(file_path):
    user_recs = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            user_id = data["custom_id"]
            recs = data["response"]["choices"][0]["message"]["content"]
            user_recs[user_id] = [x.strip() for x in recs.split("\n") if x.strip()]
    return user_recs

# Load user metadata
def load_user_metadata(file_path):
    return pd.read_csv(file_path)

# Compute SNSR (Statistical Parity in Recommendations)
def compute_snsr(user_recs, user_metadata, sensitive_attr):
    movie_groups = defaultdict(lambda: defaultdict(int))  # {movie: {group: count}}

    for user_id, recs in user_recs.items():
        group = user_metadata.loc[user_metadata["user_id"] == int(user_id), sensitive_attr].values
        if len(group) == 0:
            continue
        group = group[0]
        for movie in recs:
            movie_groups[movie][group] += 1

    # Normalize by group size
    group_counts = user_metadata[sensitive_attr].value_counts().to_dict()
    snsr_scores = {}
    for movie, counts in movie_groups.items():
        proportions = {g: counts.get(g, 0) / group_counts.get(g, 1) for g in group_counts}
        max_p = max(proportions.values())
        min_p = min(proportions.values())
        snsr_scores[movie] = max_p - min_p  # Difference in proportions

    return pd.Series(snsr_scores).sort_values(ascending=False)

# Compute SNSV (Value-based fairness - e.g., difference in average rating across groups)
def compute_snsv(user_recs, user_metadata, ratings_df, sensitive_attr):
    movie_values = defaultdict(lambda: defaultdict(list))  # {movie: {group: [ratings]}}

    for user_id, recs in user_recs.items():
        group = user_metadata.loc[user_metadata["user_id"] == int(user_id), sensitive_attr].values
        if len(group) == 0:
            continue
        group = group[0]
        for movie in recs:
            rating = ratings_df.loc[
                (ratings_df["user_id"] == int(user_id)) & (ratings_df["movie"] == movie), "rating"
            ]
            if not rating.empty:
                movie_values[movie][group].append(rating.values[0])

    snsv_scores = {}
    for movie, group_ratings in movie_values.items():
        avg_ratings = {g: sum(r)/len(r) if len(r) > 0 else 0 for g, r in group_ratings.items()}
        if len(avg_ratings) < 2:
            continue
        max_v = max(avg_ratings.values())
        min_v = min(avg_ratings.values())
        snsv_scores[movie] = max_v - min_v

    return pd.Series(snsv_scores).sort_values(ascending=False)

# Visualization helper
def plot_fairness(scores, title):
    scores.head(20).plot(kind='barh', figsize=(10, 6))
    plt.xlabel("Fairness Gap")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Main pipeline
def main():
    user_recs = load_results(OUTPUT_FILE)
    user_metadata = load_user_metadata(USER_METADATA_FILE)

    # Optionally: load rating data
    ratings_df = pd.read_csv("user_movie_ratings.csv")  # Columns: user_id, movie, rating

    print("[INFO] Calculating SNSR...")
    snsr = compute_snsr(user_recs, user_metadata, sensitive_attr="gender")
    plot_fairness(snsr, "Top 20 Movies with Highest SNSR (Gender)")

    print("[INFO] Calculating SNSV...")
    snsv = compute_snsv(user_recs, user_metadata, ratings_df, sensitive_attr="gender")
    plot_fairness(snsv, "Top 20 Movies with Highest SNSV (Gender)")

if __name__ == "__main__":
    main()
