import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from codecarbon import EmissionsTracker
import joblib
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, ENERGY_LOG_DIR, LABEL_REAL, LABEL_FAKE


def train_detector():
    print("\n=== Starting Green AI Forensic Training (with Cross-Validation) ===")
    feature_file = PROCESSED_DATA_DIR / "features.csv"
    if not feature_file.exists():
        print(f"Error: features.csv not found at {feature_file}")
        return

    df = pd.read_csv(feature_file)

    # --- 1. DATA PREPARATION ---
    df_real = df[df['label'] == LABEL_REAL]
    df_fake = df[df['label'] == LABEL_FAKE]

    # Ensure equal balance
    min_len = min(len(df_real), len(df_fake))
    print(f"Loaded Data: {len(df_real)} Real | {len(df_fake)} Fake")
    print(f"Balancing dataset to {min_len} per class...")

    df_balanced = pd.concat([
        df_real.sample(min_len, random_state=42),
        df_fake.sample(min_len, random_state=42)
    ])

    # Drop metadata columns to get feature matrix X
    X_df = df_balanced.drop(columns=['label', 'filename'])
    X = X_df.values
    y = df_balanced['label'].values

    # Get feature names automatically from the dataframe
    feature_names = X_df.columns.tolist()

    # --- 2. 5-FOLD CROSS VALIDATION (Scientific Proof) ---
    print("\nrunning 5-Fold Cross-Validation (Please wait)...")
    # We use a pipeline to ensure scaling happens INSIDE each fold (correct data science practice)
    # Using Random Forest for CV as it's generally more robust and faster for this type of data
    cv_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))

    # Run 5 folds
    # Using n_jobs=1 to avoid potential issues with CodeCarbon in subprocesses
    cv_scores = cross_val_score(cv_pipeline, X, y, cv=5, scoring='accuracy', n_jobs=1)

    print(f"Fold Scores: {cv_scores}")
    print(f" Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # --- 3. FINAL TRAINING (For Plots & Saving) ---
    print("\nTraining Final Model for Plots & Export...")

    # Split: 80% Train, 20% Test (Standard for visualization)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tracker = EmissionsTracker(output_dir=str(ENERGY_LOG_DIR), project_name="final_training_cv")
    tracker.start()

    # Train the final Random Forest model
    # This model will be used for plots and saved for other scripts
    rf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    rf_pipeline.fit(X_train, y_train)

    emissions = tracker.stop()
    print(f"Training Complete. Emissions: {emissions:.6f} kg CO2")

    # --- 4. SAVE THE MODEL ---
    # Ensure the models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the trained pipeline (including the scaler) as final_model.pkl
    model_path = MODELS_DIR / "final_model.pkl"
    joblib.dump(rf_pipeline, model_path)
    print(f"Saved final model pipeline to {model_path}")

    # --- 5. VISUALIZATION & REPORTING ---
    best_model = rf_pipeline
    acc = best_model.score(X_test, y_test)
    print(f"Note: plotting Random Forest (Test Acc: {acc:.2f})")

    # A. Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test,
        display_labels=['Real', 'Fake'], cmap=plt.cm.Blues, normalize=None
    )
    plt.title(f"Confusion Matrix (Test Acc: {acc:.2%})")
    plt.savefig(PROCESSED_DATA_DIR / "confusion_matrix.png")
    print("Saved confusion_matrix.png")

    # B. ROC Curve
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc="lower right")
    plt.title("ROC Curve")
    plt.savefig(PROCESSED_DATA_DIR / "roc_curve.png")
    print("Saved roc_curve.png")

    # C. Feature Importance
    # Extract the RF model from the pipeline to get feature importances
    rf_model = best_model.named_steps['randomforestclassifier']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(12, 6))
    plt.title("Top 10 Spectral Features")
    # Use the automatically retrieved feature names
    plt.bar(range(10), importances[indices], align="center", color='#2c7bb6')
    plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / "feature_importance.png")
    print("Saved feature_importance.png")

    print("\n--- Classification Report (Test Set) ---")
    print(classification_report(y_test, best_model.predict(X_test), target_names=['Real', 'Fake']))


if __name__ == "__main__":
    train_detector()