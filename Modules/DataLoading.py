# section1_preprocess.py
import pandas as pd
import json
import os
from datetime import datetime

# Configuration
DATA_DIR = 'Dataset/'
OUTPUT_SHORT = 'Modules/prompts/prompts_short.jsonl'
OUTPUT_FULL = 'Modules/prompts/prompts_full.jsonl'
N_PROFILES = 10
TESTMODE = True

# Load MovieLens dataset
def load_data(test_mode=False):
    ratings_cols = ["UserID", "MovieID", "Rating", "Timestamp"]
    users_cols = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
    movies_cols = ["MovieID", "Title", "Genres"]

    ratings = pd.read_csv(DATA_DIR + "ratings.dat", sep="::", names=ratings_cols, engine='python', encoding='ISO-8859-1', nrows=5000 if test_mode else None)
    users = pd.read_csv(DATA_DIR + "users.dat", sep="::", names=users_cols, engine='python', encoding='ISO-8859-1')
    movies = pd.read_csv(DATA_DIR + "movies.dat", sep="::", names=movies_cols, engine='python', encoding='ISO-8859-1')

    def age_to_group(age):
        if age < 18: return 'Teen'
        elif 18 <= age <= 35: return 'Young'
        else: return 'Adult'

    users['AgeGroup'] = users['Age'].apply(age_to_group)
    merged = pd.merge(pd.merge(ratings, movies, on='MovieID'), users, on='UserID')
    return merged, users, movies

# Temporal split
def temporal_split(data):
    train, test = [], []
    for user_id, group in data.groupby('UserID'):
        sorted_group = group.sort_values('Timestamp')
        split_idx = int(0.8 * len(sorted_group))
        train.append(sorted_group.iloc[:split_idx])
        test.append(sorted_group.iloc[split_idx:])
    return pd.concat(train), pd.concat(test)

# User profiles
def create_user_profiles(train_data, n_profiles=10):
    profiles = {}
    for user_id, group in train_data.groupby('UserID'):
        profiles[user_id] = {
            'random': group.sample(n=min(n_profiles, len(group))),
            'top-rated': group.nlargest(n_profiles, 'Rating'),
            'recent': group.nlargest(n_profiles, 'Timestamp')
        }
    return profiles

# ... [previous code remains the same]

def generate_prompt(user_info, profile_df, strategy, include_sensitive=False):
    genres = profile_df['Genres'].str.split('|').explode().value_counts().index[:3].tolist()
    years = profile_df['Title'].str.extract(r'\((\d{4})\)')[0].dropna().unique()

    sensitive = f"The user is a {user_info['AgeGroup']} {user_info['Gender']}.\n" if include_sensitive else ""
    consumed_movies = [
        f"{row['Title']} ({row['Title'][-5:-1]}, Genres: {row['Genres']}, Rating: {row['Rating']}/5)"
        for _, row in profile_df.iterrows()
    ]

    return (
        f"{sensitive}"
        f"The user mostly likes {', '.join(genres)} movies from {min(years)} to {max(years)}.\n"
        f"They recently watched: {', '.join(consumed_movies[:3])}\n"
        "Recommend 10 movies they would enjoy. Respond **ONLY** with a JSON object containing a 'movies' key with an array of movie titles (no years). Example: {\"movies\": [\"The Lion King\"]}"
    )

            
# Write prompts to JSONL file
def write_jsonl(data, filepath):
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

# Run pipeline
def main():
    full_data, users_df, movies_df = load_data(TESTMODE)
    train_df, test_df = temporal_split(full_data)
    profiles = create_user_profiles(train_df, N_PROFILES)

    all_users = users_df['UserID'].unique()
    selected_users = all_users[:2] if TESTMODE else all_users
    prompts_data = []

    for user_id in selected_users:
        user_info = users_df[users_df['UserID'] == user_id].iloc[0]
        for strategy in ['random', 'top-rated', 'recent']:
            profile_df = profiles[user_id][strategy]

            for prompt_type, sensitive in [('neutral', False), ('sensitive', True)]:
                prompt = generate_prompt(user_info, profile_df, strategy, include_sensitive=sensitive)
                prompts_data.append({
                    "custom_id": f"{user_id}_{strategy}_{prompt_type}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",  # Use model supporting JSON mode
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "response_format": {"type": "json_object"}
                    }
                })

    short_data = prompts_data[:10]  # test subset
    write_jsonl(short_data, OUTPUT_SHORT)
    write_jsonl(prompts_data, OUTPUT_FULL)
    print(f"[INFO] Generated {len(prompts_data)} prompts. Saved to {OUTPUT_SHORT} and {OUTPUT_FULL}")

if __name__ == "__main__":
    main()
