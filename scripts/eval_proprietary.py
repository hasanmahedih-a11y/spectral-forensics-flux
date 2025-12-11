import sys
import cv2
import numpy as np
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import MODELS_DIR, RAW_DATA_DIR, ANALYSIS_RESOLUTION, RAPS_SIZE
from src.features.fourier_ops import get_raps, get_spectral_slope
from src.features.bispectrum_ops import get_bispectrum_features
from src.features.cepstrum_ops import get_cepstrum_features


def extract_features(img):

    raps = get_raps(img)[:RAPS_SIZE]
    slope = get_spectral_slope(raps)
    bispec = get_bispectrum_features(img)
    cepstrum = get_cepstrum_features(img)
    return np.concatenate([raps, bispec, cepstrum, [slope]]).reshape(1, -1)


def eval_folder(model, folder_name):
    dir_path = RAW_DATA_DIR / folder_name
    images = list(dir_path.glob("*.png"))

    if not images:
        print(f" No images found in {folder_name}")
        return

    pred_fake = 0
    pred_real = 0
    total = 0

    print(f"\n Evaluating {folder_name} ({len(images)} images)...")
    for p in images:
        try:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, ANALYSIS_RESOLUTION)

            feats = extract_features(img)
            prediction = model.predict(feats)[0]

            # 1 = Fake, 0 = Real
            if prediction == 1:
                pred_fake += 1
            else:
                pred_real += 1
            total += 1
        except:
            continue

    # Calculate percentages
    fake_rate = (pred_fake / total) * 100 if total > 0 else 0
    real_rate = (pred_real / total) * 100 if total > 0 else 0

    print(f"  > Total Processed: {total}")
    print(f"  > Classified as FAKE: {pred_fake} ({fake_rate:.1f}%)")
    print(f"  > Classified as REAL: {pred_real} ({real_rate:.1f}%)")

    if fake_rate < 50:
        print(f"  Insight: The model thinks these {folder_name} images are REAL.")
    else:
        print(f"  Insight: The model correctly detects these as FAKE.")


def main():
    print("\n" + "=" * 50)
    print(" PROPRIETARY MODEL ANALYSIS")
    print("=" * 50)

    model_path = MODELS_DIR / "final_model.pkl"
    if not model_path.exists():
        print("Error: Model not found.")
        return

    model = joblib.load(model_path)

    eval_folder(model, "dalle3")
    eval_folder(model, "midjourney_v6")


if __name__ == "__main__":
    main()