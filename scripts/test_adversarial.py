import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from tqdm import tqdm
from contextlib import contextmanager

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, ANALYSIS_RESOLUTION, LABEL_REAL, LABEL_FAKE, RAPS_SIZE
from src.utils.paths import get_image_paths
from src.features.fourier_ops import get_raps, get_spectral_slope  # <--- Added Import
from src.features.bispectrum_ops import get_bispectrum_features
from src.features.cepstrum_ops import get_cepstrum_features


@contextmanager
def suppress_stderr():
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr_fd = os.dup(2)
        os.dup2(null_fd, 2)
        yield
    except OSError:
        yield
    finally:
        try:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)
            os.close(null_fd)
        except OSError:
            pass


def apply_blur(img, kernel_size):
    if kernel_size <= 1: return img
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def apply_noise(img, sigma):
    if sigma <= 0: return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def apply_median(img, kernel_size):
    if kernel_size <= 1: return img
    return cv2.medianBlur(img, kernel_size)


def extract_features(img):
    raps = get_raps(img)[:RAPS_SIZE]
    slope = get_spectral_slope(raps)  # <--- Added Slope
    bispec = get_bispectrum_features(img)
    cepstrum = get_cepstrum_features(img)
    return np.concatenate([raps, bispec, cepstrum, [slope]])  # <--- Added Slope to list


def run_attack_suite():
    print("\n=== Starting Adversarial Counter-Forensics Test ===")

    model_path = MODELS_DIR / "final_model.pkl"
    if not model_path.exists():
        print("Model not found.")
        return
    model = joblib.load(model_path)

    real_paths = get_image_paths("real")[:100]
    fake_paths = (get_image_paths("flux1") + get_image_paths("sd_v1_5"))[:100]
    all_paths = real_paths + fake_paths
    y_true = [LABEL_REAL] * len(real_paths) + [LABEL_FAKE] * len(fake_paths)

    experiments = {
        "Gaussian Blur": ([1, 3, 5, 7], apply_blur),
        "Gaussian Noise": ([0, 5, 10, 20], apply_noise),
        "Median Filter": ([1, 3, 5], apply_median)
    }

    results = {}

    for attack_name, (levels, attack_func) in experiments.items():
        print(f"\nTesting {attack_name}...")
        accuracies = []

        for level in levels:
            correct = 0
            count = 0
            with suppress_stderr():
                for i, p in enumerate(all_paths):
                    try:
                        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                        if img is None: continue
                        img = cv2.resize(img, ANALYSIS_RESOLUTION)
                        img_attacked = attack_func(img, level)

                        feats = extract_features(img_attacked)
                        # Fix shape for single prediction
                        feats = feats.reshape(1, -1)

                        pred = model.predict(feats)[0]
                        if pred == y_true[i]: correct += 1
                        count += 1
                    except:
                        continue

            acc = correct / count if count > 0 else 0
            accuracies.append(acc)
            print(f"  Level {level}: Accuracy = {acc:.2%}")

        results[attack_name] = (levels, accuracies)

    plt.figure(figsize=(10, 6))
    for name, (levels, accs) in results.items():
        plt.plot(levels, accs, marker='o', linewidth=2, label=name)

    plt.title("Robustness Against Counter-Forensic Attacks")
    plt.xlabel("Attack Intensity")
    plt.ylabel("Detection Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.0)

    out_path = PROCESSED_DATA_DIR / "adversarial_robustness.png"
    plt.savefig(out_path)
    print(f"\n Plot saved to {out_path}")


if __name__ == "__main__":
    run_attack_suite()