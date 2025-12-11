
import sys
import cv2
import numpy as np
import joblib
import torch
import warnings
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from diffusers import StableDiffusionPipeline

# --- FIX: Suppress messy warnings for a clean report ---
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import RAW_DATA_DIR, MODELS_DIR, ANALYSIS_RESOLUTION
from src.features.fourier_ops import get_raps
from src.features.bispectrum_ops import get_bispectrum_features
from src.features.cepstrum_ops import get_cepstrum_features

# --- CONFIGURATION ---
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SD_OUTPUT_DIR = RAW_DATA_DIR / "sd_v1_5"

# --- CUSTOM COUNTS ---
NUM_GEN_IMAGES = 500  # Target for AI (SD 1.5)
NUM_COCO_IMAGES = 200  # Target for Real (COCO)


def extract_features(img):
    """Helper to extract the 133 spectral features."""
    raps = get_raps(img)[:128]
    bispec = get_bispectrum_features(img)
    cepstrum = get_cepstrum_features(img)
    return np.concatenate([raps, bispec, cepstrum]).reshape(1, -1)


def run_robustness_suite():
    print("\n" + "=" * 60)
    print("  STARTING FINAL ROBUSTNESS SUITE")
    print(f"   - AI Gen Target: {NUM_GEN_IMAGES}")
    print(f"   - COCO Target:   {NUM_COCO_IMAGES}")
    print("=" * 60)

    # 1. LOAD DETECTOR
    model_path = MODELS_DIR / "tabular_svm.joblib"
    if not model_path.exists():
        print(" Error: Model not found. Run training first.")
        return
    clf = joblib.load(model_path)
    print(" Loaded Green AI Detector.")

    # ====================================================
    # TEST 1: AI GENERALIZATION (Stable Diffusion v1.5)
    # ====================================================
    print("\n--- [Test 1/2] AI Generalization (SD v1.5) ---")

    SD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = list(SD_OUTPUT_DIR.glob("*.png"))

    if len(existing_files) < NUM_GEN_IMAGES:
        print(f"Generating remaining images (Current: {len(existing_files)})...")

        pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, use_safetensors=True)
        pipe.to("cpu")
        pipe.safety_checker = None

        # Use a generic prompt to test robustness
        prompt = "A photograph of an astronaut riding a horse"

        for i in range(NUM_GEN_IMAGES):
            filename = SD_OUTPUT_DIR / f"sd15_{i:04d}.png"
            if filename.exists(): continue

            image = pipe(prompt, num_inference_steps=20).images[0]
            image.save(filename)
            print(f"Generated {i + 1}/{NUM_GEN_IMAGES}", end="\r")

        print("\nGeneration complete.")
        del pipe
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print(f"Found {len(existing_files)} images. Skipping generation.")

    # Evaluation
    sd_detected = 0
    sd_total = 0
    sd_paths = list(SD_OUTPUT_DIR.glob("*.png"))[:NUM_GEN_IMAGES]

    for p in tqdm(sd_paths, desc="Testing SD v1.5"):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, ANALYSIS_RESOLUTION)

        feats = extract_features(img)
        if clf.predict(feats)[0] == 1:  # 1 = Fake
            sd_detected += 1
        sd_total += 1

    sd_accuracy = (sd_detected / sd_total) * 100 if sd_total > 0 else 0

    # ====================================================
    # TEST 2: REAL-WORLD SAFETY (COCO Dataset)
    # ====================================================
    print("\n--- [Test 2/2] Real-World Safety (COCO) ---")

    coco_real = 0
    coco_total = 0

    try:
        # FIX: Use 'detection-datasets/coco' to avoid script error
        print("Streaming COCO (detection-datasets/coco)...")
        dataset = load_dataset("detection-datasets/coco", split="train", streaming=True)

        for i, sample in tqdm(enumerate(dataset), total=NUM_COCO_IMAGES, desc="Testing COCO"):
            if i >= NUM_COCO_IMAGES: break

            try:
                pil_image = sample['image']
                if pil_image.mode != 'RGB': continue

                img_rgb = np.array(pil_image)
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                img_resized = cv2.resize(img_gray, ANALYSIS_RESOLUTION)

                feats = extract_features(img_resized)
                # We want Prediction == 0 (Real)
                if clf.predict(feats)[0] == 0:
                    coco_real += 1
                coco_total += 1
            except:
                continue

    except Exception as e:
        print(f" COCO Error: {e}")

    coco_accuracy = (coco_real / coco_total) * 100 if coco_total > 0 else 0
    false_positive_rate = 100 - coco_accuracy

    # ====================================================
    # FINAL REPORT
    # ====================================================
    print("\n" + "=" * 60)
    print(f" FINAL ROBUSTNESS REPORT")
    print("=" * 60)
    print(f"{'Metric':<35} | {'Result':<10} | {'Interpretation'}")
    print("-" * 75)

    # Report 1
    interp_sd = "High Generalization" if sd_accuracy > 70 else "Architecture Specific"
    print(f"{'SD v1.5 Detection Rate':<35} | {sd_accuracy:.1f}%      | {interp_sd}")

    # Report 2
    interp_coco = "Safe (Low FP)" if coco_accuracy > 90 else "High False Alarms"
    print(f"{'COCO Real Accuracy':<35} | {coco_accuracy:.1f}%      | {interp_coco}")
    print("-" * 75)
    print(f"False Positive Rate (COCO): {false_positive_rate:.1f}% (Lower is better)")
    print("=" * 60)


if __name__ == "__main__":
    run_robustness_suite()