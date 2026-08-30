import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

df = pd.read_csv('data/Mobile Reviews Sentiment.csv', encoding='utf-8-sig', low_memory=False)
df_clean = df.dropna(how='all').drop_duplicates().reset_index(drop=True)
pair = df_clean[['review_text', 'rating']].dropna()

print('=== DATASET TEXT QUALITY DIAGNOSIS ===\n')
print(f'Clean rows           : {len(df_clean):,}')
print(f'Rows with review_text: {len(pair):,}')
print(f'Unique review texts  : {pair["review_text"].nunique():,}')
print(f'Duplication factor   : {len(pair) / pair["review_text"].nunique():.1f}x average repeats per text\n')

print('--- Unique texts per rating level ---')
for rv in sorted(pair['rating'].unique()):
    sub = pair[pair['rating'] == rv]
    print(f'  Rating {rv}: {len(sub):,} rows, {sub["review_text"].nunique():,} unique texts')

print()
# Check how many texts appear at MULTIPLE rating levels
text_ratings = pair.groupby('review_text')['rating'].nunique()
multi_rating = text_ratings[text_ratings > 1]
print(f'--- Text ↔ Rating overlap ---')
print(f'Texts that appear at exactly 1 rating level : {(text_ratings == 1).sum():,}')
print(f'Texts that appear at 2+ rating levels       : {len(multi_rating):,}')
print(f'Texts appearing at ALL 5 rating levels      : {(text_ratings == 5).sum():,}')
print()

# Centroid cosine similarities (to confirm collapse)
import json, numpy as np
from scipy.spatial.distance import cosine as cosine_distance
with open('memory.json', encoding='utf-8') as f:
    mem = json.load(f)
profiles = mem.get('text_rating_patterns', {}).get('rating_profiles', {})
print('--- Pairwise centroid cosine similarities ---')
keys = sorted(profiles.keys(), key=float)
centroids = {k: np.array(profiles[k]['centroid']) for k in keys}
for i, k1 in enumerate(keys):
    for k2 in keys[i+1:]:
        sim = 1.0 - cosine_distance(centroids[k1], centroids[k2])
        print(f'  Rating {k1} ↔ Rating {k2}: {sim:.4f}')

print()
print('--- Conclusion ---')
print('If centroids are all >0.99 similar: texts are essentially identical across all rating levels.')
print('This means the dataset text is NOT discriminative for rating prediction.')
print('The rating label is effectively random w.r.t. the review text.')
