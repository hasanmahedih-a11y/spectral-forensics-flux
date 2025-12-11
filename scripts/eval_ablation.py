import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Add project root
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DATA_DIR, LABEL_REAL, LABEL_FAKE

def run_ablation():
    print("\n--- Starting Ablation Study ---")
    feature_file = PROCESSED_DATA_DIR / "features.csv"
    if not feature_file.exists():
        print("Error: features.csv not found.")
        return

    df = pd.read_csv(feature_file)

    # Balance Data
    df_real = df[df['label'] == LABEL_REAL]
    df_fake = df[df['label'] == LABEL_FAKE]
    min_len = min(len(df_real), len(df_fake))
    df = pd.concat([df_real.sample(min_len, random_state=42), df_fake.sample(min_len, random_state=42)])
    y = df['label']

    # Define Groups
    feature_groups = {
        "RAPS Only": [c for c in df.columns if "feat_" in c and int(c.split('_')[1]) < 128],
        "Bispectrum Only": [c for c in df.columns if "feat_" in c and 128 <= int(c.split('_')[1]) <= 130],
        "Cepstrum Only": [c for c in df.columns if "feat_" in c and int(c.split('_')[1]) > 130],
        "Combined (All)": [c for c in df.columns if "feat_" in c]
    }

    print(f"{'Feature Set':<20} | {'Accuracy':<10}")
    print("-" * 35)
    for name, cols in feature_groups.items():
        X = df[cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        print(f"{name:<20} | {accuracy_score(y_test, clf.predict(X_test)):.4f}")

if __name__ == "__main__":
    run_ablation()