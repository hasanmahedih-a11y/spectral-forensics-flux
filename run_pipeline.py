
import sys
import os
import subprocess
import time
from pathlib import Path
import warnings

# ==========================================
#  IGNORE WARNINGS
# ==========================================
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import transformers

    transformers.logging.set_verbosity_error()
except ImportError:
    pass

try:
    import diffusers

    diffusers.logging.set_verbosity_error()
except ImportError:
    pass
# ==========================================

# MASTER SCRIPT EXECUTION LIST
# The logical order of the scientific process
SCRIPTS = [
    # --- PHASE 1: DATA ACQUISITION ---
    "scripts/download_real_data.py",  # 1. Get COCO (Real)
    "scripts/generate_flux_images.py",  # 2. Get SDXL (Fake)
    "scripts/download_proprietary.py",  # 3. Get DALL-E 3 & Midjourney

    # --- PHASE 2: FEATURE ENGINEERING ---
    "scripts/make_features.py",  # 4. Extract RAPS/Bispec/Cepstrum
    "scripts/optimize_features.py",  # 5. Run PCA/Selection Analysis

    # --- PHASE 3: MODEL TRAINING ---
    "scripts/train_tabular_detector.py",  # 6. Train Random Forest & Save Model

    # --- PHASE 4: SCIENTIFIC VALIDATION ---
    "scripts/eval_ablation.py",  # 7. Prove RAPS is key
    "scripts/visualize_spectrum.py",  # 8. Generate "Physics Proof" Plot

    # --- PHASE 5: ROBUSTNESS & SECURITY ---
    "scripts/eval_robustness.py",  # 9. Test SD 1.5 & COCO Safety
    "scripts/eval_proprietary.py",  # 10. Test DALL-E 3 & Midjourney
    "scripts/test_adversarial.py",  # 11. Test Blur/Noise Attacks
    "scripts/run_jpeg_robustness.py",  # 12. Test Compression Resilience

    # --- PHASE 6: COMPARATIVE ANALYSIS ---
    "scripts/train_baseline_cnn.py",  # 13. Train ResNet-50 (The "Red AI" Baseline)
    "scripts/plot_green_ai_comparison.py",  # 14. Generate Energy Comparison Plot

    # --- PHASE 7: OPTIMIZATION CONCLUSION ---
    "scripts/test_optimized_logic.py"  # 15. Final "Perfect" RAPS-only model

]


def run_script(script_path_str):
    """Runs a python script as a subprocess and waits for it to finish."""
    script_name = Path(script_path_str).name
    print(f"\n{'=' * 70}")
    print(f"  STARTING STEP: {script_name}")
    print(f"{'=' * 70}\n")

    start_time = time.time()
    python_exe = sys.executable

    try:
        subprocess.run([python_exe, script_path_str], check=True)
        elapsed = time.time() - start_time
        print(f"\n  FINISHED: {script_name} (Time: {elapsed:.1f}s)")

    except subprocess.CalledProcessError as e:
        print(f"\n  CRITICAL ERROR executing {script_name} (Exit Code: {e.returncode})")
        print("Pipeline stopped immediately to prevent cascading failures.")
        # We exit with 1 so you know something failed
        sys.exit(1)
    except Exception as e:
        print(f"\n  UNEXPECTED ERROR running {script_name}: {e}")
        sys.exit(1)


def main():
    print("\n" + "#" * 70)
    print("  STARTING GREEN AI FORENSIC PIPELINE (COMPLETE SUITE)")
    print("    Environment: " + sys.executable)
    print("#" * 70)

    pipeline_start_time = time.time()
    project_root = Path(__file__).parent.resolve()

    # Ensure we are working from the project root
    os.chdir(project_root)

    # 1. Pre-flight Check
    print("\n Pre-flight check: Verifying existence of all scripts...")
    missing_scripts = []
    for script_rel_path in SCRIPTS:
        full_path = project_root / script_rel_path
        if not full_path.exists():
            missing_scripts.append(str(script_rel_path))

    if missing_scripts:
        print("\n ERROR: The following necessary scripts were not found:")
        for s in missing_scripts:
            print(f"   - {s}")
        print("\n Please ensure you have created all the new scripts before running.")
        sys.exit(1)
    print(" All scripts verified. Starting execution sequence.")

    # 2. Execution Loop
    for script_rel_path in SCRIPTS:
        full_path = project_root / script_rel_path
        run_script(str(full_path))

    # 3. Final Summary
    total_time = time.time() - pipeline_start_time
    print("\n" + "#" * 70)
    print(f"  PIPELINE COMPLETE! (Total Time: {total_time / 60:.1f} minutes)")
    print("#" * 70)
    print("\nYour Final Results are available in 'data/processed/':")
    print(" 1. spectral_comparison_fixed.png (Physics)")
    print(" 2. confusion_matrix.png          (Accuracy)")
    print(" 3. roc_curve.png                 (Performance)")
    print(" 4. feature_importance.png        (Explainability)")
    print(" 5. robustness_jpeg.png           (Reliability)")
    print(" 6. adversarial_robustness.png    (Security)")
    print(" 7. green_ai_comparison.png       (Sustainability)")
    print("\n All numbers are in the logs above.")


if __name__ == "__main__":
    main()