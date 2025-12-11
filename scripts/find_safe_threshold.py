import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve

# Add project root
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, LABEL_REAL, LABEL_FAKE


def optimize_threshold(target_fpr=0.01):
    print(f"\n=== Optimizing Threshold for {target_fpr * 100}% FPR ===")

    # Load Data & Model
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    model = joblib.load(MODELS_DIR / "final_model.pkl")

    X = df.drop(columns=['label', 'filename']).values
    y = df['label'].values

    # Get Probabilities (Confidence scores)
    # prob_fake is the probability that the image is Class 1 (Fake)
    probs_fake = model.predict_proba(X)[:, 1]

    # Calculate ROC Curve
    fpr, tpr, thresholds = roc_curve(y, probs_fake, pos_label=LABEL_FAKE)

    # Find the threshold where FPR is closest to target (e.g., 1%)
    # We want FPR <= target_fpr
    valid_indices = np.where(fpr <= target_fpr)[0]

    if len(valid_indices) == 0:
        print("Could not achieve target FPR with current model.")
        return

    # Get the "safest" index (highest True Positive Rate allowed)
    best_idx = valid_indices[-1]

    safe_threshold = thresholds[best_idx]
    current_fpr = fpr[best_idx]
    current_tpr = tpr[best_idx]

    print(f" Optimal Threshold Found: {safe_threshold:.4f}")
    print(f"   - Expected False Positive Rate: {current_fpr * 100:.2f}%")
    print(f"   - Expected Detection Rate (Recall): {current_tpr * 100:.2f}%")
    print("\nRecommendation for Paper:")
    print(
        f'"By adjusting the decision threshold to {safe_threshold:.2f}, we reduced the False Positive Rate to {current_fpr * 100:.1f}% while maintaining a Detection Rate of {current_tpr * 100:.1f}%."')


if __name__ == "__main__":
    optimize_threshold(target_fpr=0.01)  # Target 1% False Alarms