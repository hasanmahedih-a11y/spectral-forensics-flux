import os
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import RAW_DATA_DIR

# CONFIGURATION
DATASET_ID = "detection-datasets/coco"
OUTPUT_DIR = RAW_DATA_DIR / "real"
NUM_IMAGES = 1000


def download_real_images():
    print(f"Streaming first {NUM_IMAGES} images from {DATASET_ID}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Stream dataset
    try:
        dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    except Exception as e:
        print(f"Critical Error: {e}")
        return

    count = 0
    for sample in tqdm(dataset, total=NUM_IMAGES):
        if count >= NUM_IMAGES: break

        try:
            image = sample['image']
            if image.mode != 'RGB': image = image.convert('RGB')
            if image.size[0] < 256 or image.size[1] < 256: continue

            filename = OUTPUT_DIR / f"real_{count:04d}.png"
            if not filename.exists():
                image.save(filename)
            count += 1
        except:
            continue

    print(f"Successfully saved {count} diverse real images.")


if __name__ == "__main__":
    download_real_images()