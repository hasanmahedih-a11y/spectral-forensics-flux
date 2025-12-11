import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import joblib
from contextlib import contextmanager

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DATA_DIR, ANALYSIS_RESOLUTION, LABEL_REAL, LABEL_FAKE, RAPS_SIZE, MODELS_DIR
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


def extract_features_from_image(img):
    if img is None: return None
    raps = get_raps(img)[:RAPS_SIZE]
    slope = get_spectral_slope(raps)  # <--- Added Slope
    bispec = get_bispectrum_features(img)
    cepstrum = get_cepstrum_features(img)
    return np.concatenate([raps, bispec, cepstrum, [slope]])  # <--- Added Slope


def _test_images(paths, model, target_label, quality):
    correct_predictions = 0
    for p in paths:
        try:
            with suppress_stderr():
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, ANALYSIS_RESOLUTION)

            with suppress_stderr():
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encimg = cv2.imencode('.jpg', img, encode_param)
                decimg = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
            if decimg is None: continue

            feats = extract_features_from_image(decimg)
            feats = feats.reshape(1, -1)  # Fix shape

            if model.predict(feats)[0] == target_label:
                correct_predictions += 1
        except Exception:
            continue
    return correct_predictions


def run_jpeg_test(model, real_paths, fake_paths, qualities=[100, 90, 80, 70, 60, 50]):
    total_images = len(real_paths) + len(fake_paths)
    print(f"Testing on {total_images} images...")
    accuracies = []
    for q in tqdm(qualities, desc="Testing JPEG"):
        correct_real = _test_images(real_paths, model, LABEL_REAL, q)
        correct_fake = _test_images(fake_paths, model, LABEL_FAKE, q)
        acc = (correct_real + correct_fake) / total_images
        accuracies.append(acc)
        tqdm.write(f"Quality {q}: Accuracy = {acc * 100:.2f}%")
    return qualities, accuracies


def plot_results(qualities, accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(qualities, accuracies, marker='o', color='purple')
    plt.title("Detector Robustness to JPEG")
    plt.xlabel("Quality Factor")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.ylim(0.5, 1.0)
    plt.gca().invert_xaxis()
    plt.savefig(PROCESSED_DATA_DIR / "robustness_jpeg.png")
    print("Saved plot.")


def main():
    model_path = MODELS_DIR / "final_model.pkl"
    if not model_path.exists(): return
    model = joblib.load(model_path)
    subset_size = 100
    real_paths = get_image_paths("real")[:subset_size]
    fake_paths = (get_image_paths("flux1") + get_image_paths("sd_v1_5"))[:subset_size]
    q, a = run_jpeg_test(model, real_paths, fake_paths)
    plot_results(q, a)


if __name__ == "__main__":
    main()

