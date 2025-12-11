import os
from pathlib import Path

# --- Project Paths ---
# Dynamically find the root of the project relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ENERGY_LOG_DIR = PROJECT_ROOT / "energy_logs"

# Ensure logs directory exists
os.makedirs(ENERGY_LOG_DIR, exist_ok=True)

# --- Data Constants ---
# Flux generates at 1024x1024, but we analyze at 256x256
# to standardize frequency bins (Source: Section 4.1)
GEN_RESOLUTION = (1024, 1024)
ANALYSIS_RESOLUTION = (256, 256)

# Class Labels
LABEL_REAL = 0
LABEL_FAKE = 1

# --- Feature Extraction Hyperparameters ---
# RAPS (Radially Averaged Power Spectrum)
RAPS_SIZE = 128  # Nyquist limit for 256px image

# Bispectrum Block Processing (Source: Section 4.2.2)
BISPECTRUM_BLOCK_SIZE = 32
BISPECTRUM_OVERLAP = 0.5

# Cepstrum Analysis
CEPSTRUM_FILTER_RADIUS = 10