import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DATA_DIR, LABEL_REAL, LABEL_FAKE


def optimize_features():
    print("\n=== STARTING FEATURE OPTIMIZATION ===")

    # 1. Load Data
    feature_file = PROCESSED_DATA_DIR / "features.csv"
    if not feature_file.exists():
        print("Error: features.csv not found.")
        return
    df = pd.read_csv(feature_file)

    # Balance Data (Same logic as main training)
    df_real = df[df['label'] == LABEL_REAL]
    df_fake = df[df['label'] == LABEL_FAKE]
    min_len = min(len(df_real), len(df_fake))
    df_balanced = pd.concat([
        df_real.sample(min_len, random_state=42),
        df_fake.sample(min_len, random_state=42)
    ])

    X = df_balanced.drop(columns=['label', 'filename']).values
    y = df_balanced['label'].values

    print(f"Data Loaded: {len(X)} samples, {X.shape[1]} features.")

    # 2. Define Baselines
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # --- EXPERIMENT A: Baseline (All Features) ---
    pipe_base = make_pipeline(StandardScaler(), rf)
    scores_base = cross_val_score(pipe_base, X, y, cv=cv, scoring='accuracy')
    print(f"\n[Baseline] All Features: {scores_base.mean():.4f} (+/- {scores_base.std() * 2:.4f})")

    # --- EXPERIMENT B: PCA (Dimensionality Reduction) ---
    # Keep 95% of variance
    pipe_pca = make_pipeline(StandardScaler(), PCA(n_components=0.95), rf)
    scores_pca = cross_val_score(pipe_pca, X, y, cv=cv, scoring='accuracy')
    print(f"[PCA] 95% Variance:     {scores_pca.mean():.4f} (+/- {scores_pca.std() * 2:.4f})")

    # Check how many components PCA kept
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95).fit(X_scaled)
    print(f"   -> Reduced features from {X.shape[1]} to {pca.n_components_} components.")

    # --- EXPERIMENT C: Mutual Information (Feature Selection) ---
    # Select top 50 features
    pipe_mi = make_pipeline(StandardScaler(), SelectKBest(mutual_info_classif, k=50), rf)
    scores_mi = cross_val_score(pipe_mi, X, y, cv=cv, scoring='accuracy')
    print(f"[Mutual Info] Top 50:     {scores_mi.mean():.4f} (+/- {scores_mi.std() * 2:.4f})")

    # --- Conclusion ---
    best_score = max(scores_base.mean(), scores_pca.mean(), scores_mi.mean())
    if scores_pca.mean() >= scores_base.mean() - 0.01:
        print("\n RECOMMENDATION: Use PCA.")
        print("It maintains accuracy while removing noise, likely improving robustness.")
    elif scores_mi.mean() > scores_base.mean():
        print("\n RECOMMENDATION: Use Mutual Information Selection.")
    else:
        print("\n RECOMMENDATION: Stick with All Features (Baseline is robust).")


if __name__ == "__main__":
    optimize_features()