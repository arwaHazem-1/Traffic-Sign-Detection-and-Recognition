"""
predict.py — Run the trained classifier on a single image.

Usage:
    python predict.py --image path/to/sign.jpg --model eval_results/classifier.pkl

The model file is created automatically by evaluation.py after training.
"""

import argparse
import sys
import logging
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog

from classification import TrafficSignClassifier
from Preprocessing import full_pipeline as preprocess_np
from main import run_pipeline

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def predict_single(image_path: str, model_path: str, full_pipeline: bool = False):
    """
    Load a trained classifier and predict the class of one image.

    Args:
        image_path   : Path to the input image file.
        model_path   : Path to the .pkl file saved by evaluation.py.
        full_pipeline: If True, also run harris/pyramid/sift/segmentation
                       and save visualisation outputs next to the image.
    """
    # ── Load image ────────────────────────────────────────────────────────
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        sys.exit(f"[ERROR] Cannot read image: {image_path}")
    log.info(f"Image loaded: {image_path}  ({img_bgr.shape[1]}×{img_bgr.shape[0]})")

    # ── Load model ────────────────────────────────────────────────────────
    classifier, class_names = TrafficSignClassifier.load(model_path)
    if not classifier.is_trained:
        sys.exit("[ERROR] Loaded model is not trained. Re-run evaluation.py first.")
    log.info(f"Model loaded: {model_path}  ({len(class_names)} classes)")

    # ── Preprocess ────────────────────────────────────────────────────────
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    prep_f32 = preprocess_np(img_rgb, size=(128, 128),
                             normalize_method="minmax",
                             equalize=True, denoise=True)
    prep_u8  = (prep_f32 * 255).astype(np.uint8)

    # ── Predict ───────────────────────────────────────────────────────────
    preds, confs = classifier.predict([prep_u8])
    idx          = int(preds[0])
    confidence   = float(confs[0])
    label        = class_names[idx] if (class_names and idx < len(class_names)) else str(idx)

    # ── Print result ──────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  PREDICTION RESULT")
    print("═" * 50)
    print(f"  Image      : {Path(image_path).name}")
    print(f"  Class index: {idx}")
    print(f"  Class name : {label}")
    print(f"  Confidence : {confidence:.4f}  ({confidence*100:.1f}%)")
    print("═" * 50)

    # ── Top-5 probabilities ───────────────────────────────────────────────
    gray = cv2.cvtColor(prep_u8, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (64, 64))
    feat = hog(gray, orientations=9, pixels_per_cell=(8,8),
               cells_per_block=(2,2), block_norm='L2-Hys',
               feature_vector=True).astype(np.float32)
    feat_scaled = classifier.scaler.transform([feat])
    all_probs   = classifier.model.predict_proba(feat_scaled)[0]

    top5_idx = np.argsort(all_probs)[::-1][:5]
    print("\n  Top-5 predictions:")
    for rank, i in enumerate(top5_idx, 1):
        name = class_names[i] if (class_names and i < len(class_names)) else str(i)
        print(f"    {rank}. {name:<35} {all_probs[i]*100:5.1f}%")
    print()

    # ── Optional full pipeline ────────────────────────────────────────────
    if full_pipeline:
        out_dir = str(Path(image_path).parent / "predict_output")
        log.info(f"Running full pipeline, saving visuals to {out_dir}")
        run_pipeline(
            img_bgr,
            classifier=classifier,
            class_names=class_names,
            output_dir=out_dir,
            image_name=Path(image_path).stem,
            verbose=True,
        )
        log.info(f"Pipeline outputs saved to {out_dir}")

    return {"label": label, "index": idx, "confidence": confidence}


def main():
    parser = argparse.ArgumentParser(
        description="Predict traffic sign class for a single image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image",  required=True,
                        help="Path to the input image.")
    parser.add_argument("--model",  required=True,
                        help="Path to classifier.pkl saved by evaluation.py.")
    parser.add_argument("--pipeline", action="store_true",
                        help="Also run the full pipeline and save stage visualisations.")
    args = parser.parse_args()

    predict_single(args.image, args.model, full_pipeline=args.pipeline)


if __name__ == "__main__":
    main()
