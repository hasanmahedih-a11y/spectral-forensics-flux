import sys
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# ==========================================
# SILENCE WARNINGS FOR THIS SUBPROCESS
# ==========================================
# This is necessary because subprocesses do not inherit warning filters
# from the parent script. This will silence the harmless sklearn warnings
# about missing feature names when predicting on numpy arrays.
warnings.filterwarnings("ignore")
# ==========================================

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, LABEL_REAL, LABEL_FAKE, RAPS_SIZE
)
from src.utils.paths import get_image_paths

# Import feature extraction logic from make_features.py to ensure consistency
sys.path.append(str(Path(__file__).parent))
from make_features import extract_features_from_image


def load_and_split_data():
    """Loads data and splits it into RAPS-only features and labels."""
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")

    # Separate features (X) and labels (y)
    X_all = df.drop(columns=['label', 'filename'])
    y = df['label']

    # Select only RAPS features (the first RAPS_SIZE columns)
    # We use iloc to select by position, which is robust.
    X_raps = X_all.iloc[:, :RAPS_SIZE]

    return X_raps, y


def train_universal_model(X, y):
    """Trains a RandomForest on the provided features."""
    # 1. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Train model
    # Using a smaller, faster model for this experiment
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_scaled, y)

    # Evaluate on training data just as a sanity check
    y_pred = clf.predict(X_scaled)
    acc = accuracy_score(y, y_pred)
    print(f"Training Accuracy (RAPS Only): {acc * 100:.2f}%")

    return clf, scaler


def test_generalization(model, scaler):
    """Tests the model on unseen SD 1.5 images."""
    print("\n[Experiment 2] Testing Generalization on SD 1.5 (with RAPS Only)...")
    sd15_paths = get_image_paths("sd_v1_5")[:500]  # Test on 500 images

    correct = 0
    total = 0

    for p in tqdm(sd15_paths):
        try:
            # 1. Extract all features
            # We use the same function from make_features.py for consistency
            # This function is already wrapped with OS-level warning suppression.
            full_feats = extract_features_from_image(p)
            if full_feats is None: continue

            # 2. Slice to get only RAPS features
            raps_feats = full_feats[:RAPS_SIZE]

            # 3. Scale & Predict
            # Reshape to (1, -1) because we are predicting on a single sample
            # This is where the harmless sklearn warning would occur without the filter.
            feats_scaled = scaler.transform(raps_feats.reshape(1, -1))
            pred = model.predict(feats_scaled)[0]

            if pred == LABEL_FAKE:
                correct += 1
            total += 1
        except Exception:
            continue

    if total > 0:
        acc = correct / total
        print(f"SD 1.5 Detection Rate: {acc * 100:.1f}%")
        if acc > 0.92:  # Compare to previous baseline of ~90-92%
            print("SUCCESS: Removing complex features IMPROVED generalization!")
        else:
            print("RESULT: Generalization is similar or slightly lower.")

    return correct, total


def test_confidence_thresholds(model, scaler):
    """Tests how changing the confidence threshold affects the False Positive Rate."""
    print("\n[Experiment 3] Testing Confidence Thresholds (Safety Check)...")
    # Test on real images to find False Positives
    real_paths = get_image_paths("real")[:500]

    probs = []
    for p in tqdm(real_paths):
        try:
            full_feats = extract_features_from_image(p)
            if full_feats is None: continue
            raps_feats = full_feats[:RAPS_SIZE]
            feats_scaled = scaler.transform(raps_feats.reshape(1, -1))
            # Get probability of being FAKE (class 1)
            prob_fake = model.predict_proba(feats_scaled)[0][1]
            probs.append(prob_fake)
        except Exception:
            continue

    probs = np.array(probs)
    total_real = len(probs)

    print("\nThreshold  | False Positive Rate ")
    print("-----------------------------------")
    for threshold in [0.5, 0.7, 0.9, 0.99]:
        # A false positive is a real image with prob_fake > threshold
        false_positives = np.sum(probs > threshold)
        fpr = false_positives / total_real
        print(f"{threshold:<10} | {fpr * 100:.1f}%")

    print("\nCONCLUSION:")
    print("If FPR drops at 0.9, suggest 'High-Confidence Thresholding' in your paper.")


def main():
    print("\n\n=== TESTING LOGIC IMPROVEMENTS ===")

    # 1. Load & Prepare Data
    print("\n[Experiment 1] Training 'Universal' RAPS-Only Model...")
    X_raps, y = load_and_split_data()

    # 2. Train the simplified model
    model, scaler = train_universal_model(X_raps, y)

    # 3. Run Experiments
    test_generalization(model, scaler)
    test_confidence_thresholds(model, scaler)


if __name__ == "__main__":
    main()