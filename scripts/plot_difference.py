import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DATA_DIR, LABEL_REAL, LABEL_FAKE, RAPS_SIZE


def plot_difference():
    print("Loading features.csv...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")

    # 1. Get RAPS columns (feat_0 to feat_127)
    raps_cols = [f"feat_{i}" for i in range(RAPS_SIZE)]

    # 2. Calculate Means
    real_mean = df[df["label"] == LABEL_REAL][raps_cols].mean().values
    fake_mean = df[df["label"] == LABEL_FAKE][raps_cols].mean().values

    # 3. Calculate Difference (The Forensic Signal)
    diff = real_mean - fake_mean

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(diff, color='purple', linewidth=2, label='Difference (Real - Fake)')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)

    # Highlight the area where Real > Fake (Positive Difference)
    plt.fill_between(range(len(diff)), diff, 0, where=(diff > 0), color='blue', alpha=0.1, label='Real has more detail')
    plt.fill_between(range(len(diff)), diff, 0, where=(diff < 0), color='red', alpha=0.1,
                     label='Fake has more artifact')

    plt.title("The Hidden Signal: Spectral Difference (Real vs. SDXL)")
    plt.xlabel("Spatial Frequency")
    plt.ylabel("Power Difference")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = PROCESSED_DATA_DIR / "spectral_difference.png"
    plt.savefig(output_path, dpi=300)
    print(f" Signal revealed! Saved to: {output_path}")


if __name__ == "__main__":
    plot_difference()