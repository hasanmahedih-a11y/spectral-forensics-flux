import torch
import os
from diffusers import AutoPipelineForText2Image
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import RAW_DATA_DIR

# --- CONFIGURATION ---
MODEL_ID = "stabilityai/sdxl-turbo"
OUTPUT_DIR = RAW_DATA_DIR / "flux1"
NUM_IMAGES = 1000

# --- DIVERSE PROMPTS (Matching COCO) ---
PROMPTS = [
    "a portrait of a person walking down a busy street, realistic lighting, 4k",
    "a shiny red sports car parked on a city street, automotive photography",
    "a bowl of fresh fruit on a wooden kitchen table, morning sunlight",
    "a cute dog playing in a park with a ball, shallow depth of field",
    "a modern living room with a sofa and television, interior design photo",
    "a landscape view of a mountain range at sunset, national geographic style",
    "a close up of a cat sleeping on a rug, highly detailed fur texture",
    "a busy intersection in tokyo at night, neon lights, street photography",
    "a plate of delicious pasta with tomato sauce in a restaurant",
    "an airplane flying in a clear blue sky, telephoto lens"
]


def generate_images():
    print(f"Loading Model: {MODEL_ID}...")

    if torch.cuda.is_available():
        pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.float16, variant="fp16").to("cuda")
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID).to("cpu")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(NUM_IMAGES):
        filename = OUTPUT_DIR / f"flux_{i:04d}.png"
        if filename.exists(): continue

        prompt = PROMPTS[i % len(PROMPTS)]
        print(f"[{i + 1}/{NUM_IMAGES}] Generating: {prompt}")
        pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0].save(filename)


if __name__ == "__main__":
    generate_images()