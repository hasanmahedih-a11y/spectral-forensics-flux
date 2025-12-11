import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from contextlib import contextmanager
from PIL import Image


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


# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DATA_DIR, ANALYSIS_RESOLUTION, RAPS_SIZE
from src.utils.paths import get_image_paths
from src.features.fourier_ops import get_raps


def safe_load_image(path):
    try:
        with Image.open(path) as pil_img:
            pil_img = pil_img.convert('L')
            pil_img = pil_img.resize(ANALYSIS_RESOLUTION)
            return np.array(pil_img)
    except Exception:
        return None


def get_spectra_data(image_paths, label_name, desc):
    spectra_list = []
    freq_bins = np.arange(RAPS_SIZE)

    print(f"  -> Found {len(image_paths)} file paths for {label_name}")

    with suppress_stderr():
        for p in tqdm(image_paths, desc=desc, leave=False):
            img = safe_load_image(p)
            if img is None: continue

            try:
                # Extract RAPS
                raps = get_raps(img)[:RAPS_SIZE]

                # --- CRITICAL FIX: LOG TRANSFORM ---
                # Convert tiny raw numbers to Decibels (Log Scale)
                # We add 1e-10 to avoid log(0) errors
                log_raps = np.log10(raps + 1e-10)

                df_temp = pd.DataFrame({
                    'Frequency Bin': freq_bins,
                    'Log Power': log_raps,  # Using Log values now!
                    'Label': label_name
                })
                spectra_list.append(df_temp)
            except Exception:
                continue

    if not spectra_list:
        print(f"  WARNING: No valid images were processed for {label_name}!")
        return pd.DataFrame()

    print(f"  -> Successfully extracted spectra from {len(spectra_list)} images.")
    return pd.concat(spectra_list, ignore_index=True)


def plot_spectral_comparison():
    print("\n=== Generating Spectral Fingerprint Plot ===")

    # Use all 1000 images for the best smooth line
    subset_size = 1000
    real_paths = get_image_paths("real")[:subset_size]

    # Combine Flux and SD1.5
    flux_paths = get_image_paths("flux1")
    sd15_paths = get_image_paths("sd_v1_5")
    fake_paths = (flux_paths + sd15_paths)[:subset_size]

    print("\nProcessing Real Images...")
    df_real = get_spectra_data(real_paths, 'Real (COCO)', "Processing Real")

    print("\nProcessing Fake Images...")
    df_fake = get_spectra_data(fake_paths, 'Synthetic (Flux/SDXL)', "Processing Fake")

    if df_real.empty or df_fake.empty:
        print(" ERROR: One of the datasets is empty. Check your data folders.")
        return

    # Combine data
    df_plot = pd.concat([df_real, df_fake], ignore_index=True)

    # Plotting
    print("\nGenerating Plot...")
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    sns.lineplot(
        data=df_plot,
        x='Frequency Bin',
        y='Log Power',
        hue='Label',
        palette={'Real (COCO)': '#1f77b4', 'Synthetic (Flux/SDXL)': '#d62728'},
        linewidth=2
    )

    # Zoom in on High Frequencies (Where the "Drop-off" happens)
    # Skip the first 20 bins (Low Freq) to see the detail
    plt.xlim(20, RAPS_SIZE - 1)

    plt.title("Spectral Divergence: Real vs. Synthetic (Log-Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("Frequency Bin (Radial Distance)", fontsize=12)
    plt.ylabel("Log Magnitude (Energy)", fontsize=12)
    plt.legend(title='Image Type', loc='upper right')
    plt.tight_layout()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "spectral_comparison_fixed.png"
    plt.savefig(output_path, dpi=300)
    print(f" Success! Fixed plot saved to {output_path}")


if __name__ == "__main__":
    plot_spectral_comparison()