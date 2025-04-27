import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURE THESE PATHS ---
DATASET_PATH = 'Dataset/full_movies_data.csv'
RESULTS_PATH = 'prompt_results/batch_680a0fd42548819084bed92a08d876f1_results.csv'
# ------------------------------

def load_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Cannot find ratings CSV at {DATASET_PATH}")
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Cannot find results CSV at {RESULTS_PATH}")

    full = pd.read_csv(DATASET_PATH)
    recs = pd.read_csv(RESULTS_PATH, header=None, names=['custom_id','movies'])
    return full, recs

def preprocess(recs):
    # parse movie lists
    recs['movies_list'] = (
        recs['movies']
        .str.split(',')
        .apply(lambda L: [m.strip().strip('"') for m in L])
    )
    # extract group & variation
    def parse_id(cid):
        # e.g. "1_recent_gender_age_only_baseline"
        suffix = cid.split('1_recent_')[-1]
        if '_' not in suffix:
            return suffix, suffix
        grp, var = suffix.rsplit('_',1)
        return grp, var

    recs[['group','variation']] = recs['custom_id'].apply(
        lambda cid: pd.Series(parse_id(cid))
    )
    return recs

def compute_popularity(full):
    if 'Title' not in full.columns:
        raise ValueError("Expected a 'Title' column in the ratings CSV.")
    # count number of ratings per title
    return full['Title'].value_counts().to_dict()

def annotate_metrics(recs, popularity):
    # avg & log-avg popularity
    recs['avg_pop'] = recs['movies_list'].apply(
        lambda lst: np.mean([popularity.get(m,0) for m in lst])
    )
    recs['log_avg_pop'] = np.log1p(recs['avg_pop'])

    # prepare neutral lookup
    neutral = (
        recs[recs['group']=='neutral']
        .set_index('variation')['movies_list']
        .to_dict()
    )

    # Jaccard similarity
    def jaccard(a,b):
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # fairness vs neutral
    def fairness(row):
        if row['group']=='neutral':
            return 1.0
        base = neutral.get(row['variation'], [])
        return jaccard(row['movies_list'], base)

    recs['fairness_jaccard'] = recs.apply(fairness, axis=1)
    return recs

def aggregate_and_save_summary(recs):
    # group-level averages
    stats = recs.groupby('group').agg(
        avg_popularity=('avg_pop','mean'),
        avg_log_popularity=('log_avg_pop','mean'),
        avg_fairness=('fairness_jaccard','mean')
    ).reset_index()

    # write text summary
    corr = recs['log_avg_pop'].corr(recs['fairness_jaccard'])
    with open('analysis_summary.txt','w') as f:
        f.write("Descriptive Analysis of LLM Recommendations\n")
        f.write("===========================================\n\n")
        f.write(f"- Total prompt variations: {len(recs)}\n")
        f.write(f"- Prompt groups: {', '.join(stats['group'])}\n\n")
        f.write("Average metrics by prompt group:\n")
        f.write(stats.to_string(index=False))
        f.write("\n\n")
        f.write(f"Pearson corr(log-avg-popularity, fairness): {corr:.3f}\n")

    return stats

def plot_and_save(stats, recs):
    # 1) avg log-pop by group
    plt.figure(figsize=(8,5))
    plt.bar(stats['group'], stats['avg_log_popularity'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Log-Popularity by Prompt Group')
    plt.tight_layout()
    plt.savefig('avg_log_popularity_by_group.png')
    plt.close()

    # 2) avg fairness by group
    plt.figure(figsize=(8,5))
    plt.bar(stats['group'], stats['avg_fairness'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Jaccard Fairness by Prompt Group')
    plt.tight_layout()
    plt.savefig('avg_fairness_by_group.png')
    plt.close()

    # 3) scatter popularity vs fairness
    plt.figure(figsize=(8,6))
    for g, grp in recs.groupby('group'):
        plt.scatter(grp['log_avg_pop'], grp['fairness_jaccard'], label=g)
    for _, row in recs.iterrows():
        plt.annotate(row['variation'],
                     (row['log_avg_pop'], row['fairness_jaccard']),
                     fontsize=8, alpha=0.7)
    plt.xlabel('Log Average Popularity')
    plt.ylabel('Jaccard Fairness vs Neutral')
    plt.title('Popularity Bias vs Fairness by Prompt Variation')
    plt.legend()
    plt.tight_layout()
    plt.savefig('popularity_vs_fairness.png')
    plt.close()

def main():
    full, recs = load_data()
    recs = preprocess(recs)
    popularity = compute_popularity(full)
    recs = annotate_metrics(recs, popularity)
    stats = aggregate_and_save_summary(recs)
    plot_and_save(stats, recs)
    print("✅ Analysis complete.")
    print("  • Summary: analysis_summary.txt")
    print("  • Charts: avg_log_popularity_by_group.png, avg_fairness_by_group.png, popularity_vs_fairness.png")

if __name__ == '__main__':
    main()
