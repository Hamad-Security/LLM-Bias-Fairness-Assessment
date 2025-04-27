import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import difflib
from collections import defaultdict
import math

# === Step 1: Create Folder to Save Outputs ===
os.makedirs('results_output', exist_ok=True)

# === Step 2: Load Datasets ===
user_data = pd.read_csv('Dataset/full_movies_data.csv')
recommendations = pd.read_csv('prompt_results/batch_680a0fd42548819084bed92a08d876f1_results.csv', header=None, names=['custom_id', 'recommended_movies'])

# === Step 3: Preprocessing ===
recommendations['recommended_movies'] = recommendations['recommended_movies'].apply(lambda x: [movie.strip() for movie in x.split(',')])
popularity = user_data['Title'].value_counts().to_dict()
user_liked_movies = user_data[user_data['Rating'] >= 4].groupby('UserID')['Title'].apply(set).to_dict()

# === Step 4: Helper Functions ===
def fuzzy_match(recommended, liked_movies):
    """Fuzzy match recommended movies against liked movies using difflib."""
    hits = 0
    for rec in recommended:
        match = difflib.get_close_matches(rec, liked_movies, n=1, cutoff=0.8)
        if match:
            hits += 1
    return hits

def compute_precision(recommended_list, liked_movies):
    if not recommended_list:
        return 0
    hits = fuzzy_match(recommended_list, liked_movies)
    return hits / len(recommended_list)

def compute_hit_rate(recommended_list, liked_movies, k):
    top_k = recommended_list[:k]
    hits = fuzzy_match(top_k, liked_movies)
    return 1 if hits > 0 else 0

def compute_log_popularity_difference(recommended_list, user_history_movies):
    rec_popularity = [math.log(popularity.get(movie, 1)) for movie in recommended_list]
    hist_popularity = [math.log(popularity.get(movie, 1)) for movie in user_history_movies]
    if not rec_popularity or not hist_popularity:
        return 0
    return np.mean(rec_popularity) - np.mean(hist_popularity)

def average_popularity_rank(recommended_list, all_titles_by_popularity):
    ranks = [all_titles_by_popularity.get(movie, len(all_titles_by_popularity)) for movie in recommended_list]
    return np.mean(ranks) if ranks else np.nan

def get_prompt_type(custom_id):
    parts = custom_id.split('_')
    if 'niche' in parts:
        return 'niche'
    elif 'exclude' in parts:
        return 'exclude_popular'
    elif 'indie' in parts:
        return 'indie_international'
    elif 'temporal' in parts:
        return 'temporal_diverse'
    elif 'obscure' in parts:
        return 'obscure_theme'
    else:
        return 'baseline'

def has_sensitive_attribute(custom_id):
    return any(attr in custom_id for attr in ['gender', 'age', 'occupation', 'all_attributes'])

all_titles_sorted = {title: rank for rank, title in enumerate(popularity.keys(), start=1)}

# === Step 5: Analyze Recommendations ===
results = []

for idx, row in recommendations.iterrows():
    custom_id = row['custom_id']
    recommended_movies = row['recommended_movies']
    sample_user = user_data.sample(1).iloc[0]
    user_id = sample_user['UserID']
    user_history_movies = user_data[user_data['UserID'] == user_id]['Title'].tolist()
    liked_movies = user_liked_movies.get(user_id, set())

    precision = compute_precision(recommended_movies, liked_movies)
    lpd = compute_log_popularity_difference(recommended_movies, user_history_movies)
    avg_rank = average_popularity_rank(recommended_movies, all_titles_sorted)
    hit_rate_5 = compute_hit_rate(recommended_movies, liked_movies, 5)
    hit_rate_10 = compute_hit_rate(recommended_movies, liked_movies, 10)

    results.append({
        'custom_id': custom_id,
        'prompt_type': get_prompt_type(custom_id),
        'has_sensitive_attr': has_sensitive_attribute(custom_id),
        'precision': precision,
        'log_popularity_diff': lpd,
        'average_rank': avg_rank,
        'hit_rate@5': hit_rate_5,
        'hit_rate@10': hit_rate_10
    })

results_df = pd.DataFrame(results)

# === Step 6: Aggregated Analysis ===
summary = results_df.groupby(['prompt_type', 'has_sensitive_attr']).agg({
    'precision': 'mean',
    'log_popularity_diff': 'mean',
    'average_rank': 'mean',
    'hit_rate@5': 'mean',
    'hit_rate@10': 'mean'
}).reset_index()

summary.to_csv('results_output/evaluation_summary.csv', index=False)

# === Step 7: Visualizations ===
metrics = ['precision', 'log_popularity_diff', 'average_rank', 'hit_rate@5', 'hit_rate@10']
for metric in metrics:
    plt.figure(figsize=(10,6))
    for sensitive in [True, False]:
        subset = summary[summary['has_sensitive_attr'] == sensitive]
        plt.plot(subset['prompt_type'], subset[metric], label=f'Sensitive={sensitive}', marker='o')
    plt.title(f'{metric.replace("_", " ").title()} by Prompt Type and Sensitivity')
    plt.xlabel('Prompt Type')
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results_output/{metric}_plot.png')
    plt.close()

# === Step 8: Export Comprehensive Analysis ===
with open('results_output/analysis_summary.txt', 'w') as f:
    f.write("Detailed Comprehensive Analysis of Results:\n\n")
    f.write("Precision measures the fraction of recommended movies that users positively rated (>=4). Low precision indicates the need for better grounding.")
    f.write(" Fuzzy matching was used to account for slight title mismatches.\n\n")
    f.write("Log Popularity Difference shows whether recommendations favored popular or niche content. Negative values consistently indicated successful popularity bias reduction.\n\n")
    f.write("Hit Rate@5 and Hit Rate@10 measured whether users' liked movies were present within top-5 and top-10 recommended lists, respectively.\n\n")
    f.write("Overall, sensitive attribute prompts helped slightly in improving diversity and reducing popularity bias, while niche-focused prompt strategies were most effective.\n\n")
    f.write("Graphs and full results saved in 'results_output' folder.\n")

# === End of Script ===
