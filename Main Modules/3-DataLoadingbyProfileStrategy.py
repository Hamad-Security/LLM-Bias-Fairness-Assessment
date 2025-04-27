"""
This script processes the MovieLens dataset to generate a series of prompts intended for a movie recommendation system API.
The prompts aim to simulate different recommendation scenarios by incorporating various user profile attributes and recommendation biases.
Below is an overview of what the script does:

1. **Dataset Loading and Preprocessing:**
   - **Datasets Used:**
     - **Ratings**: Contains movie ratings provided by users (columns: UserID, MovieID, Rating, Timestamp).
     - **Users**: Contains information about users (columns: UserID, Gender, Age, Occupation, Zip-code).
     - **Movies**: Contains movie details (columns: MovieID, Title, Genres).
   - **Preprocessing Steps:**
     - Reads the datasets with specific settings (e.g., using a custom separator `::` and specific encoding).
     - Converts the raw user age into an age group (`Teen`, `Young`, or `Adult`) using the helper function `age_to_group`.
     - Merges the datasets (ratings, movies, and users) into a single DataFrame for later processing.

2. **Temporal Split of Data:**
   - Implements a function `temporal_split` to divide the merged dataset into training and testing sets based on a temporal split.
   - For each user, their records are sorted by timestamp. Approximately the first 80% of the records per user are used for training, while the remaining 20% are reserved for testing.

3. **User Profile Creation:**
   - The function `create_user_profiles` constructs different user profiles (or subsets of the viewing history) for each user from the training data.
   - Three distinct strategies are used for creating profiles:
     - **Random**: A random sample of the user's viewing history.
     - **Top-rated**: The top-rated movies that the user has watched.
     - **Recent**: The most recent movies that the user watched.
   - The number of records per profile is limited (default is 10).

4. **Prompt Generation:**
   - The core function `generate_prompt` generates prompts for a movie recommendation system.
   - **User-Sensitive Attributes:**
     - Based on the `fairness_type` parameter, the prompt includes sensitive user information:
       - For `gender_age_only` and `all_attributes`, it adds the user's age group and gender.
       - For `occupation_only` and `all_attributes`, it adds the user's occupation (mapped from numeric codes to descriptive strings).
   - **Consumed Movies Information:**
     - The function processes the user’s movie history (extracted from one of the three profiles) and formats details like title (with the year extracted from the title), genres, and rating.
   - **Instruction Setup:**
     - The prompt starts with a role instruction declaring that the system is an expert movie recommendation engine.
     - An introduction lists a sample of movies from the user's recent viewing history.
     - Conditional bias instructions can be added based on a bias strategy (e.g., recommending niche or independent movies).
   - **Final Output:**
     - Prompts are constructed as strings that instruct the API to output a JSON object containing a list of recommended movie titles.

5. **Output Generation:**
   - The generated prompts (each including a unique `custom_id` and a configuration for a POST request to an API endpoint) are written into two JSONL files:
     - A short file containing a subset of prompts (e.g., the first 10 prompts) for testing or debugging (`Modules/prompts/prompts_short.jsonl`).
     - A full file with all generated prompts (`Modules/prompts/prompts_full.jsonl`).

6. **Execution Flow:**
   - The `main` function orchestrates the entire process:
     - Loads and preprocesses the data.
     - Splits the data temporally.
     - Creates user profiles.
     - Iterates over a set of users (a subset if in test mode) and over all possible combinations of recommendation strategies, fairness settings, and bias strategies.
     - Generates prompts and writes the output to files.
   - Finally, when executed as a standalone script (i.e., `__main__`), it runs the entire pipeline and prints a confirmation message with the total number of prompts generated.

This comprehensive setup is designed to facilitate robust testing of the movie recommendation system, ensuring that various fairness and bias parameters are systematically explored through multiple prompt variations.
"""

import pandas as pd
import json
import os
import re
from datetime import datetime

# Configuration
DATA_DIR = 'Dataset/'
OUTPUT_SHORT = 'Main Modules/prompts/prompts_short.jsonl'
OUTPUT_FULL = 'Main Modules/prompts/prompts_full.jsonl'
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
        if age < 18:
            return 'Teen'
        elif 18 <= age <= 35:
            return 'Young'
        else:
            return 'Adult'

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
''
# Revised prompt generation function
def generate_prompt(user_info, profile_df, strategy, fairness_type='neutral', bias_strategy=None):
    # Gender mapping: 'F' -> 'female', 'M' -> 'male'
    gender_map = {'F': 'female', 'M': 'male'}
    gender_str = gender_map.get(user_info['Gender'], user_info['Gender'])
    
    # Build sensitive context if applicable
    sensitive_parts = []
    if fairness_type in ['gender_age_only', 'all_attributes']:
        sensitive_parts.append(f"{user_info['AgeGroup']} {gender_str}")
    if fairness_type in ['occupation_only', 'all_attributes']:
        occupation_map = {
            0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin", 4: "college/grad student",
            5: "customer service", 6: "doctor/health care", 7: "executive/managerial", 8: "farmer",
            9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired", 
            14: "sales/marketing", 15: "scientist", 16: "self-employed", 17: "technician/engineer", 
            18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
        }
        occupation = occupation_map.get(user_info['Occupation'], 'unspecified')
        sensitive_parts.append(occupation)
    sensitive = f"The user is a {' '.join(sensitive_parts)}.\n" if sensitive_parts else ""
    
    # Process consumed movies without repeating year info
    consumed_movies = []
    for _, row in profile_df.iterrows():
        # Extract year from title using regex
        match = re.search(r'\((\d{4})\)', row['Title'])
        if match:
            year = match.group(1)
            base_title = row['Title'].replace(f" ({year})", "")
        else:
            year = "unknown"
            base_title = row['Title']
        consumed_movies.append(f"{base_title} ({year}, Genres: {row['Genres']}, Rating: {row['Rating']}/5)")
    
    # Role instruction and introduction
    role_instruction = "You are an expert movie recommendation system. "
    introduction = "The user's recent viewing history is: "
    
    # Bias instruction based on selected strategy.
    bias_instruction = ""
    if bias_strategy == "niche_genre":
        bias_instruction = "Focus on recommending movies from less common or niche genres that the user has engaged with."
    elif bias_strategy == "exclude_popular":
        bias_instruction = "Avoid recommending popular or blockbuster movies."
    elif bias_strategy == "indie_international":
        bias_instruction = "Recommend independent or international movies, steering clear of major studio releases."
    elif bias_strategy == "temporal_diverse":
        bias_instruction = "Encourage recommendations that span a wider temporal range, including older films."
    elif bias_strategy == "obscure_theme":
        bias_instruction = "Recommend underrated and lesser-known films that match the unique aspects of the user's taste."
    
    # Construct the final prompt
    prompt = (
        f"{role_instruction}\n"
        f"{sensitive}"
        f"{introduction}{', '.join(consumed_movies[:3])}.\n"
        f"{bias_instruction}\n"
        "Recommend 10 movies they would enjoy. Respond **ONLY** with a JSON object containing a 'movies' key with an array of movie titles (no years). "
        "Example: {\"movies\": [\"The Lion King\"]}"
    )
    
    return prompt

''# Write prompts to JSONL file
def write_jsonl(data, filepath):
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
def main():
    full_data, users_df, movies_df = load_data(TESTMODE)
    train_df, test_df = temporal_split(full_data)
    profiles = create_user_profiles(train_df, N_PROFILES)

    all_users = users_df['UserID'].unique()
    selected_users = all_users[:2] if TESTMODE else all_users

    fairness_options = ['neutral', 'gender_age_only', 'occupation_only', 'all_attributes']
    bias_strategies = [None, "niche_genre", "exclude_popular", "indie_international", "temporal_diverse", "obscure_theme"]
    strategies = ['random', 'top-rated', 'recent']

    # Prepare output directories if they don't exist
    output_dir = "Main Modules/prompts/"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare containers for each strategy
    strategy_outputs = {strategy: [] for strategy in strategies}

    for user_id in selected_users:
        user_info = users_df[users_df['UserID'] == user_id].iloc[0]
        for strategy in strategies:
            profile_df = profiles[user_id][strategy]
            for fairness_type in fairness_options:
                for bias_strategy in bias_strategies:
                    prompt = generate_prompt(user_info, profile_df, strategy, fairness_type, bias_strategy)
                    strategy_label = bias_strategy if bias_strategy else "baseline"
                    custom_id = f"{user_id}_{strategy}_{fairness_type}_{strategy_label}"
                    strategy_outputs[strategy].append({
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4.1-nano",
                            "temperature": 0.7,
                            "messages": [
                                {"role": "user", "content": prompt}
                            ]
                        }
                    })

    # Write outputs for each strategy
    for strategy, data in strategy_outputs.items():
        output_path = f"{output_dir}prompts_{strategy}.jsonl"
        write_jsonl(data, output_path)
        print(f"Wrote {len(data)} prompts to {output_path}")

    print("Prompt generation completed.")

if __name__ == "__main__":
    main()
