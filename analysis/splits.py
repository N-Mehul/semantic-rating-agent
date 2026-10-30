import os
import json
import hashlib
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import joblib

DATA_PATH = os.path.join("data", "actual_reviews.xlsx")
RESULTS_DIR = "results"
SEED = 42

def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # strip quotes
    text = re.sub(r'^["\']|["\']$', '', text.strip())
    # collapse internal whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=== Dataset Lineage Verification ===")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return
        
    df = pd.read_excel(DATA_PATH, header=None)
    # The columns from batch_predict.py loader
    cols_25 = [
        "review_id", "name", "age", "brand", "model", "price_usd",
        "price_local", "currency", "exchange_rate_to_usd", "rating",
        "review_text", "sentiment", "country", "language", "date",
        "verified_purchase", "battery_life_rating", "camera_rating",
        "performance_rating", "design_rating", "display_rating",
        "review_length", "word_count", "helpful_votes", "source"
    ]
    if len(df.columns) == 25:
        df.columns = cols_25
    else:
        print(f"Warning: Expected 25 columns, got {len(df.columns)}")
        # Assuming column 10 is review_text and 9 is rating if not named
        if 'review_text' not in df.columns:
            df = df.rename(columns={10: 'review_text', 9: 'rating'})
            
    total_rows = len(df)
    print(f"Total row count: {total_rows} (Expected: ~10,003)")
    
    # Rating range and counts
    print("Rating distribution:")
    rating_counts = df['rating'].value_counts().sort_index()
    print(rating_counts)
    
    # Normalization
    df['raw_text'] = df['review_text'].astype(str)
    df['norm_text'] = df['raw_text'].apply(normalize_text)
    df['text_hash'] = df['norm_text'].apply(hash_text)
    
    raw_unique = df['raw_text'].nunique()
    norm_unique = df['norm_text'].nunique()
    
    print("\n=== Unique Text Counts ===")
    print(f"Raw unique text count: {raw_unique}")
    print(f"Normalized unique text count: {norm_unique} (Expected: ~110)")
    if raw_unique != norm_unique:
        print(f"Note: Normalization collapsed {raw_unique} raw texts into {norm_unique} normalized groups.")
    
    # Embeddings cache
    print("\n=== Caching Embeddings ===")
    cache_path = os.path.join(RESULTS_DIR, "embedding_cache.npz")
    if not os.path.exists(cache_path):
        from sentence_transformers import SentenceTransformer
        # Use the same model as in agent.py (all-MiniLM-L6-v2)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        unique_texts = df['norm_text'].unique()
        print(f"Embedding {len(unique_texts)} unique texts...")
        embeddings = model.encode(unique_texts, show_progress_bar=True)
        embed_dict = {hash_text(text): emb for text, emb in zip(unique_texts, embeddings)}
        np.savez(cache_path, **embed_dict)
        print(f"Saved embedding cache to {cache_path}")
    else:
        print(f"Embedding cache already exists at {cache_path}")
        
    # Splits
    print("\n=== Building Splits ===")
    indices = np.arange(total_rows)
    
    # Regime A: Row-level split
    train_idx_A, temp_idx_A = train_test_split(indices, test_size=0.30, random_state=SEED)
    val_idx_A, test_idx_A = train_test_split(temp_idx_A, test_size=0.50, random_state=SEED)
    
    # Check overlap in A
    train_hashes_A = set(df.loc[train_idx_A, 'text_hash'])
    test_hashes_A = set(df.loc[test_idx_A, 'text_hash'])
    overlap_A = len([h for h in df.loc[test_idx_A, 'text_hash'] if h in train_hashes_A])
    print(f"Regime A (Row-level): {overlap_A} out of {len(test_idx_A)} test rows overlap with train texts.")
    
    # Save A
    with open(os.path.join(RESULTS_DIR, "splits_regimeA.json"), "w") as f:
        json.dump({
            "train": train_idx_A.tolist(),
            "val": val_idx_A.tolist(),
            "test": test_idx_A.tolist()
        }, f)
        
    # Regime B: Grouped split
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx_B, temp_idx_B = next(gss1.split(df, groups=df['text_hash']))
    
    # further split temp into val and test
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    temp_df = df.iloc[temp_idx_B]
    val_idx_temp, test_idx_temp = next(gss2.split(temp_df, groups=temp_df['text_hash']))
    
    val_idx_B = temp_idx_B[val_idx_temp]
    test_idx_B = temp_idx_B[test_idx_temp]
    
    # Check overlap in B
    train_hashes_B = set(df.loc[train_idx_B, 'text_hash'])
    test_hashes_B = set(df.loc[test_idx_B, 'text_hash'])
    overlap_B = len([h for h in df.loc[test_idx_B, 'text_hash'] if h in train_hashes_B])
    print(f"Regime B (Grouped): {overlap_B} out of {len(test_idx_B)} test rows overlap with train texts.")
    
    # Save B
    with open(os.path.join(RESULTS_DIR, "splits_regimeB.json"), "w") as f:
        json.dump({
            "train": train_idx_B.tolist(),
            "val": val_idx_B.tolist(),
            "test": test_idx_B.tolist()
        }, f)
        
    print("Splits persisted.")

if __name__ == "__main__":
    main()
