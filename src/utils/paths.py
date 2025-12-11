import sys
from pathlib import Path
from typing import List

# Add project root to python path to ensure imports work if run directly
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import RAW_DATA_DIR


def get_image_paths(source: str) -> List[Path]:
    """
    Retrieves all .png image paths for a specific source directory.

    Args:
        source (str): The subdirectory name in data/raw (e.g., 'real', 'flux1', 'sdxl').

    Returns:
        List[Path]: A list of pathlib.Path objects pointing to the images.
    """
    target_dir = RAW_DATA_DIR / source

    if not target_dir.exists():
        print(f"Warning: Directory {target_dir} does not exist.")
        return []

    # We strictly look for .png to avoid JPEG compression artifacts
    # Source: Section 5, Phase 1
    images = list(target_dir.glob("*.png"))

    if not images:
        print(f"Warning: No .png images found in {target_dir}")

    return images


if __name__ == "__main__":
    # Quick test to see if it works
    print(f"Checking for 'real' images in: {RAW_DATA_DIR / 'real'}")
    paths = get_image_paths("real")
    print(f"Found {len(paths)} images.")