import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from collections import defaultdict
import re
from sklearn.preprocessing import MultiLabelBinarizer

# Configuration
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
plt.style.use('ggplot')

# Load datasets
movies_df = pd.read_csv('Dataset/full_movies_data.csv')
recs_df = pd.read_csv('prompt_results/batch_680a0fd42548819084bed92a08d876f1_results.csv')

# Preprocessing functions
def clean_title(title):
    """Remove year and special characters from titles"""
    return re.sub(r'\s*\(\d{4}\)', '', title).strip()

def parse_genres(genre_str):
    """Convert genre string to list"""
    return genre_str.split('|') if isinstance(genre_str, str) else []

# Preprocess movies data
movies_df['clean_title'] = movies_df['Title'].apply(clean_title)
movies_df['Genres'] = movies_df['Genres'].apply(parse_genres)

# Create movie metadata dataframe
movie_metadata = movies_df.groupby('clean_title').agg({
    'MovieID': 'first',
    'Genres': 'first',
    'Rating': ['mean', 'count']
}).reset_index()
movie_metadata.columns = ['title', 'movie_id', 'genres', 'avg_rating', 'rating_count']
movie_metadata['popularity'] = np.log1p(movie_metadata['rating_count'])  # Log-normalized popularity

# Preprocess recommendations data
def parse_recommendations(recommendations):
    try:
        return [clean_title(title.strip()) for title in recommendations.split(',')]
    except:
        return []

recs_df['recommendations'] = recs_df.iloc[:, 1].apply(parse_recommendations)
recs_df[['user_group', 'strategy_type']] = recs_df['custom_id'].str.extract(r'^(\d+_.+?)_(baseline|niche_genre|exclude_popular|indie_international|temporal_diverse|obscure_theme)$')

# Expand recommendations into individual rows
exploded_recs = recs_df.explode('recommendations').merge(
    movie_metadata, 
    left_on='recommendations', 
    right_on='title', 
    how='left'
)

# Analysis 1: Popularity Bias Analysis
popularity_analysis = exploded_recs.groupby(['custom_id', 'user_group', 'strategy_type']).agg({
    'popularity': 'mean',
    'rating_count': 'mean',
    'avg_rating': 'mean'
}).reset_index()

# Analysis 2: Genre Analysis
exploded_recs['genres'] = exploded_recs['genres'].apply(
    lambda x: x if isinstance(x, list) else []
)

# Modified Genre Analysis
exploded_genres = exploded_recs.explode('genres')

# Count genre occurrences per recommendation strategy
genre_counts = exploded_genres.groupby(['custom_id', 'genres']).size().unstack(fill_value=0)

# Calculate genre percentages
genre_percentages = genre_counts.div(genre_counts.sum(axis=1), axis=0)

# Filter out empty genres if any
genre_percentages = genre_percentages.drop(columns=[''], errors='ignore')

# Analysis 3: Demographic Fairness
def extract_demographics(user_group):
    demographics = {}
    parts = user_group.split('_')
    demographics['age_group'] = parts[1] if 'recent' not in parts else 'neutral'
    demographics['gender'] = next((p for p in parts if p in ['neutral', 'gender', 'male', 'female']), 'neutral')
    demographics['occupation'] = next((p for p in parts if p in ['occupation', 'student']), 'neutral')
    return pd.Series(demographics)

demographic_data = recs_df['user_group'].apply(extract_demographics)
demographic_analysis = pd.concat([recs_df[['custom_id', 'strategy_type']], demographic_data], axis=1)

# Analysis 4: Temporal Analysis (Year-based)
movies_df['year'] = movies_df['Title'].str.extract(r'\((\d{4})\)').astype(float)
movie_years = movies_df.groupby('clean_title')['year'].first()
exploded_recs = exploded_recs.merge(movie_years, left_on='title', right_index=True, how='left')

# Visualization 1: Popularity Comparison
plt.figure(figsize=(14, 8))
sns.boxplot(x='strategy_type', y='popularity', hue='user_group', data=popularity_analysis)
plt.title('Popularity Distribution by Recommendation Strategy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 2: Genre Distribution Heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(genre_percentages.T, cmap='viridis', 
           annot=True, fmt=".1%", 
           cbar_kws={'label': 'Genre Percentage'})
plt.title('Genre Distribution Across Recommendation Strategies')
plt.xlabel('Recommendation Strategy')
plt.ylabel('Genres')
plt.tight_layout()
plt.show()

# Visualization 3: Temporal Distribution
plt.figure(figsize=(14, 8))
sns.swarmplot(x='strategy_type', y='year', hue='user_group',
             data=exploded_recs, dodge=True, palette="dark")
plt.title('Release Year Distribution by Recommendation Strategy')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Visualization 4: Demographic Fairness
demo_metrics = demographic_analysis.merge(
    popularity_analysis, 
    on=['custom_id', 'user_group', 'strategy_type']
)

plt.figure(figsize=(14, 8))
sns.catplot(x='strategy_type', y='popularity', 
           hue='gender', col='age_group',
           data=demo_metrics, kind='box',
           height=6, aspect=0.8)
plt.suptitle('Popularity Distribution by Demographic Groups')
plt.tight_layout()
plt.show()

# Advanced Analysis: Recommendation Diversity
def calculate_diversity(recommendation_list):
    genres = exploded_recs[exploded_recs['recommendations'].isin(recommendation_list)]['genres']
    all_genres = [g for sublist in genres for g in sublist]
    return len(set(all_genres)) / len(all_genres) if all_genres else 0

recs_df['diversity'] = recs_df['recommendations'].apply(calculate_diversity)

# Visualization 5: Diversity Analysis
plt.figure(figsize=(14, 8))
sns.barplot(x='strategy_type', y='diversity', hue='user_group', data=recs_df)
plt.title('Genre Diversity by Recommendation Strategy')
plt.xticks(rotation=45)
plt.ylabel('Diversity Index (Unique Genres/Total)')
plt.tight_layout()
plt.show()

# Statistical Testing
from scipy import stats

def perform_anova_test(data, metric):
    groups = [group[metric].values for name, group in data.groupby('strategy_type')]
    f_val, p_val = stats.f_oneway(*groups)
    print(f'ANOVA results for {metric}: F={f_val:.2f}, p={p_val:.4f}')

perform_anova_test(popularity_analysis, 'popularity')
perform_anova_test(recs_df, 'diversity')

# Export Analysis Results
analysis_results = {
    'popularity_analysis': popularity_analysis,
    'genre_analysis': genre_percentages,
    'demographic_analysis': demographic_analysis,
    'diversity_analysis': recs_df[['custom_id', 'diversity']]
}

with pd.ExcelWriter('recommendation_analysis.xlsx') as writer:
    for sheet_name, df in analysis_results.items():
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)