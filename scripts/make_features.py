import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from codecarbon import EmissionsTracker
from contextlib import contextmanager
from PIL import Image, ImageFile

# Ensure truncated images don't crash the script
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==============================================================================
#  OS-LEVEL WARNING SUPPRESSION
# ==============================================================================
@contextmanager
def suppress_stderr():
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield;
        return
    try:
        old_stderr_fd = os.dup(2)
    except OSError:
        os.close(null_fd);
        yield;
        return
    try:
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        os.close(null_fd)


# ==============================================================================

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import (
    PROCESSED_DATA_DIR,
    ANALYSIS_RESOLUTION,
    LABEL_REAL,
    LABEL_FAKE,
    RAPS_SIZE,
    ENERGY_LOG_DIR
)
from src.utils.paths import get_image_paths
# UPGRADE: Added get_spectral_slope import
from src.features.fourier_ops import get_raps, get_spectral_slope
from src.features.bispectrum_ops import get_bispectrum_features
from src.features.cepstrum_ops import get_cepstrum_features


def extract_features_from_image(image_path):
    """
    UPGRADE: Loads image using Pillow (safer than cv2),
    and extracts RAPS, Slope, Bispectrum, and Cepstrum.
    """
    try:
        # 1. Safer Loading with Pillow
        # (Fixes libpng warnings and "missing real data" issues)
        with Image.open(image_path) as pil_img:
            pil_img = pil_img.convert('L')  # Grayscale
            pil_img = pil_img.resize(ANALYSIS_RESOLUTION)
            img = np.array(pil_img)
    except Exception:
        return None

    # 2. Extract Features
    raps = get_raps(img)[:RAPS_SIZE]

    # UPGRADE: Calculate Spectral Slope
    slope = get_spectral_slope(raps)

    bispec = get_bispectrum_features(img)
    cepstrum = get_cepstrum_features(img)

    # 3. Combine (Add slope to the end)
    return np.concatenate([raps, bispec, cepstrum, [slope]])


def process_dataset():
    # Setup
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ENERGY_LOG_DIR.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(output_dir=str(ENERGY_LOG_DIR), project_name="feature_extraction_ensemble")
    tracker.start()

    all_data = []

    print("Loading file paths...")
    real_paths = get_image_paths("real")

    # Ensemble: Flux + SD 1.5
    flux_paths = get_image_paths("flux1")
    sd15_paths = get_image_paths("sd_v1_5")
    fake_paths = flux_paths + sd15_paths

    print(f"Found {len(real_paths)} Real images.")
    print(f"Found {len(fake_paths)} Fake images (Mixed Flux + SD1.5).")

    # Process Real
    # Using suppress_stderr just in case, though Pillow is usually clean
    with suppress_stderr():
        for p in tqdm(real_paths, desc="Processing Real"):
            feats = extract_features_from_image(p)
            if feats is not None:
                row = {'label': LABEL_REAL, 'filename': p.name}
                for i, val in enumerate(feats): row[f'feat_{i}'] = val
                all_data.append(row)

    # Process Fake
    with suppress_stderr():
        for p in tqdm(fake_paths, desc="Processing Fake (Ensemble)"):
            feats = extract_features_from_image(p)
            if feats is not None:
                row = {'label': LABEL_FAKE, 'filename': p.name}
                for i, val in enumerate(feats): row[f'feat_{i}'] = val
                all_data.append(row)

    tracker.stop()

    # Save
    df = pd.DataFrame(all_data)
    output_path = PROCESSED_DATA_DIR / "features.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved upgraded features (with Slope) to {output_path}")


if __name__ == "__main__":
    process_dataset()