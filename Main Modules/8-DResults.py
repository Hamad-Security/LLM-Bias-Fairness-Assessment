import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import difflib
from collections import defaultdict
import math

# === Step 1: Create Folder to Save Outputs ===
os.makedirs('results_output_8', exist_ok=True)

# === Step 2: Load Datasets ===
user_data = pd.read_csv('Dataset/full_movies_data.csv')
recommendations = pd.read_csv('LlmCsvOutput/merged_prompt_results_top_rated.csv', header=None, names=['custom_id', 'recommended_movies'])

# === Step 3: Preprocessing ===
recommendations['recommended_movies'] = recommendations['recommended_movies'].apply(lambda x: [movie.strip() for movie in x.split(',')])
popularity = user_data['Title'].value_counts().to_dict()
user_liked_movies = user_data[user_data['Rating'] >= 3].groupby('UserID')['Title'].apply(set).to_dict()

# === Step 4: Helper Functions ===
def fuzzy_match(recommended, liked_movies):
    """Fuzzy match recommended movies against liked movies using difflib."""
    hits = 0
    for rec in recommended:
        match = difflib.get_close_matches(rec, liked_movies, n=1, cutoff=0.6)
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

# ... (Previous imports and steps 1-4 remain unchanged)

# === Modified Helper Functions ===
def parse_custom_id(custom_id):
    """Properly parse custom_id into components"""
    parts = custom_id.split('_')
    user_id = parts[0]
    strategy = parts[1]
    fairness_type = parts[2]
    bias_designator = '_'.join(parts[3:])  # Handle multi-part bias designators
    return user_id, strategy, fairness_type, bias_designator

# === Updated Analysis Loop ===
results = []

for idx, row in recommendations.iterrows():
    custom_id = row['custom_id']
    recommended_movies = row['recommended_movies']
    
    try:
        # Parse custom ID components
        user_id, strategy, fairness_type, bias_designator = parse_custom_id(custom_id)
        
        # Get user-specific data
        liked_movies = user_liked_movies.get(int(user_id), set())
        user_history_movies = user_data[user_data['UserID'] == int(user_id)]['Title'].tolist()

        # Calculate metrics
        precision = compute_precision(recommended_movies, liked_movies)
        lpd = compute_log_popularity_difference(recommended_movies, user_history_movies)
        avg_rank = average_popularity_rank(recommended_movies, all_titles_sorted)
        hit_rate_5 = compute_hit_rate(recommended_movies, liked_movies, 5)
        hit_rate_10 = compute_hit_rate(recommended_movies, liked_movies, 10)

    except (IndexError, ValueError) as e:
        print(f"Skipping malformed custom_id: {custom_id} - Error: {str(e)}")
        continue

    results.append({
        'strategy': strategy,
        'fairness_type': fairness_type,
        'bias_designator': bias_designator,
        'precision': precision,
        'log_popularity_diff': lpd,
        'average_rank': avg_rank,
        'hit_rate@5': hit_rate_5,
        'hit_rate@10': hit_rate_10
    })

# ... (Remaining code for analysis and visualizations remains unchanged)
# === Updated Analysis Loop ===
results = []

for idx, row in recommendations.iterrows():
    custom_id = row['custom_id']
    recommended_movies = row['recommended_movies']
    
    # Parse custom ID components
    try:
        strategy, fairness_type, bias_designator = parse_custom_id(custom_id)
    except IndexError:
        print(f"Skipping malformed custom_id: {custom_id}")
        continue

    # ... (Metric calculations remain unchanged)

    results.append({
        'strategy': strategy,
        'fairness_type': fairness_type,
        'bias_designator': bias_designator,
        'precision': precision,
        'log_popularity_diff': lpd,
        'average_rank': avg_rank,
        'hit_rate@5': hit_rate_5,
        'hit_rate@10': hit_rate_10
    })

results_df = pd.DataFrame(results)

# === Enhanced Aggregation ===
# Create multiple analysis dimensions
strategy_fairness = results_df.groupby(['strategy', 'fairness_type']).mean(numeric_only=True).reset_index()
bias_fairness = results_df.groupby(['bias_designator', 'fairness_type']).mean(numeric_only=True).reset_index()
strategy_bias = results_df.groupby(['strategy', 'bias_designator']).mean(numeric_only=True).reset_index()

# === Enhanced Visualizations ===
import seaborn as sns

# 1. Heatmap Matrix
metrics = ['precision', 'log_popularity_diff', 'hit_rate@5']
for metric in metrics:
    plt.figure(figsize=(12, 8))
    pivot = strategy_fairness.pivot_table(index='strategy', 
                                        columns='fairness_type', 
                                        values=metric)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f'{metric.title()} by Strategy and Fairness Type')
    plt.savefig(f'results_output_8/heatmap_{metric}_strategy_fairness.png')
    plt.close()

# 2. Multi-line Plots by Fairness Type
plt.figure(figsize=(14, 8))
for fairness_type in results_df['fairness_type'].unique():
    subset = strategy_bias[strategy_bias['bias_designator'] == fairness_type]
    plt.plot(subset['strategy'], subset['log_popularity_diff'], 
            label=fairness_type, marker='o')

plt.title('Popularity Bias Reduction by Strategy and Fairness Type')
plt.xlabel('Recommendation Strategy')
plt.ylabel('Log Popularity Difference')
plt.legend(title='Fairness Type')
plt.grid(True)
plt.savefig('results_output_8/popularity_bias_strategy_fairness.png')
plt.close()

# 3. Faceted Histograms
g = sns.FacetGrid(results_df, col='bias_designator', hue='fairness_type', 
                 col_wrap=3, height=4)
g.map(sns.barplot, 'strategy', 'precision', order=['random', 'top-rated', 'recent'])
g.add_legend()
g.savefig('results_output_8_8/faceted_precision_analysis.png')

# === Enhanced Tables ===
# Generate detailed comparison tables
comparison_tables = {
    'strategy_fairness': pd.pivot_table(results_df, 
                                      index='strategy', 
                                      columns='fairness_type',
                                      values=metrics,
                                      aggfunc='mean'),
    
    'bias_metrics': pd.pivot_table(results_df,
                                 index='bias_designator',
                                 columns='fairness_type',
                                 values=['log_popularity_diff', 'average_rank'],
                                 aggfunc='mean')
}

# Save tables as CSV and HTML
for name, table in comparison_tables.items():
    table.to_csv(f'results_output_8/{name}_comparison.csv')
    table.to_html(f'results_output_8/{name}_comparison.html')

# === Updated Text Analysis ===
# ... (Existing text analysis remains but updates metrics references)