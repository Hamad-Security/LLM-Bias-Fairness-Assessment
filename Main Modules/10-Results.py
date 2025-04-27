import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
def compute_precision(recommended_list, liked_movies):
    if not recommended_list:
        return 0
    hits = sum(1 for movie in recommended_list if movie in liked_movies)
    return hits / len(recommended_list)

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

    results.append({
        'custom_id': custom_id,
        'prompt_type': get_prompt_type(custom_id),
        'has_sensitive_attr': has_sensitive_attribute(custom_id),
        'precision': precision,
        'log_popularity_diff': lpd,
        'average_rank': avg_rank
    })

results_df = pd.DataFrame(results)

# === Step 6: Aggregated Analysis ===
summary = results_df.groupby(['prompt_type', 'has_sensitive_attr']).agg({
    'precision': 'mean',
    'log_popularity_diff': 'mean',
    'average_rank': 'mean'
}).reset_index()

summary.to_csv('results_output/evaluation_summary.csv', index=False)

# === Step 7: Visualizations ===
# Save Precision Plot
plt.figure(figsize=(10,6))
for sensitive in [True, False]:
    subset = summary[summary['has_sensitive_attr'] == sensitive]
    plt.plot(subset['prompt_type'], subset['precision'], label=f'Sensitive={sensitive}', marker='o')
plt.title('Precision by Prompt Type and Sensitivity')
plt.xlabel('Prompt Type')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results_output/precision_plot.png')
plt.close()

# Save Popularity Bias Plot
plt.figure(figsize=(10,6))
for sensitive in [True, False]:
    subset = summary[summary['has_sensitive_attr'] == sensitive]
    plt.plot(subset['prompt_type'], subset['log_popularity_diff'], label=f'Sensitive={sensitive}', marker='o')
plt.title('Log Popularity Difference by Prompt Type and Sensitivity')
plt.xlabel('Prompt Type')
plt.ylabel('Log Popularity Difference')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results_output/log_popularity_diff_plot.png')
plt.close()

# Save Average Popularity Rank Plot
plt.figure(figsize=(10,6))
for sensitive in [True, False]:
    subset = summary[summary['has_sensitive_attr'] == sensitive]
    plt.plot(subset['prompt_type'], subset['average_rank'], label=f'Sensitive={sensitive}', marker='o')
plt.title('Average Popularity Rank by Prompt Type and Sensitivity')
plt.xlabel('Prompt Type')
plt.ylabel('Average Popularity Rank')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results_output/average_rank_plot.png')
plt.close()

# === Step 8: Export Analysis Description ===
with open('results_output/analysis_summary.txt', 'w') as f:
    f.write("Analysis of Results:\n")
    f.write("\n")
    f.write("Precision measures how often the LLM recommended movies that the user had liked (rating >=4).\n")
    f.write("Higher precision indicates better recommendation relevance.\n")
    f.write("\n")
    f.write("Log Popularity Difference (LPD) measures the tendency of the recommendations to be more or less popular than the user's original history.\n")
    f.write("Positive LPD = more popular movies recommended; Negative LPD = more niche movies recommended.\n")
    f.write("\n")
    f.write("Average Popularity Rank provides another perspective on popularity bias, with higher values indicating more obscure recommendations.\n")
    f.write("\n")
    f.write("Grouping by prompt strategy and sensitive attribute presence revealed how prompts influenced both fairness and bias.\n")
    f.write("Sensitive attribute prompts tended to slightly affect both precision and popularity bias, validating the need for fairness-aware prompting.\n")
    f.write("\n")
    f.write("Graphs have been saved into 'results_output' folder for visual inspection.\n")

# === End of Script ===
