import os
import sys
import io
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import RAW_DATA_DIR

# --- CONFIGURATION ---
# 1. DALL-E 3 Source
DALLE_ID = "OpenDatasets/dalle-3-dataset"
# 2. Midjourney V6 Source
MJ_ID = "brivangl/midjourney-v6-llava"


def download_set(dataset_id, folder_name, num_images=200):
    save_dir = RAW_DATA_DIR / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStreaming {num_images} images from {dataset_id} -> {folder_name}...")

    try:
        # Load dataset in streaming mode
        dataset = load_dataset(dataset_id, split="train", streaming=True)

        count = 0
        for i, sample in enumerate(tqdm(dataset)):
            if count >= num_images: break

            try:
                # 1. Try to get the image object
                img_data = sample.get('image') or sample.get('jpg')

                # 2. Handle bytes vs PIL object
                if isinstance(img_data, bytes):
                    img = Image.open(io.BytesIO(img_data))
                elif isinstance(img_data, dict) and 'bytes' in img_data:
                    img = Image.open(io.BytesIO(img_data['bytes']))
                else:
                    img = img_data

                if img is None: continue

                # 3. Validation & Save
                if img.mode != 'RGB': img = img.convert('RGB')

                # Resize if massive to save space, but keep high quality
                if img.width > 1024:
                    img = img.resize((1024, 1024))

                img.save(save_dir / f"{folder_name}_{count:03d}.png")
                count += 1

            except Exception:
                continue

        print(f" Saved {count} images to {save_dir}")

    except Exception as e:
        print(f" Error downloading {dataset_id}: {e}")


if __name__ == "__main__":
    # Download DALL-E 3
    download_set(DALLE_ID, "dalle3", 200)

    # Download Midjourney V6
    download_set(MJ_ID, "midjourney_v6", 200)
