"""
evaluation.py — Quantitative Evaluation with confusion matrix and segmentation IoU
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

from main import run_pipeline
from classification import TrafficSignClassifier
from segmentation import segment_traffic_signs

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  IoU HELPERS
# ─────────────────────────────────────────────

def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Intersection-over-Union between two binary masks.

    Args:
        pred_mask : Predicted binary mask (uint8 or bool, any non-zero = foreground).
        gt_mask   : Ground-truth binary mask (same shape).

    Returns:
        IoU score in [0, 1]. Returns 0.0 if both masks are empty.
    """
    pred_bool = pred_mask.astype(bool)
    gt_bool   = gt_mask.astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union        = np.logical_or(pred_bool,  gt_bool).sum()

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_box_iou(box_a: Tuple[int,int,int,int],
                    box_b: Tuple[int,int,int,int]) -> float:
    """
    Compute IoU between two bounding boxes in (x, y, w, h) format.

    Returns:
        IoU score in [0, 1].
    """
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = box_a[0] + box_a[2], box_a[1] + box_a[3]

    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = box_b[0] + box_b[2], box_b[1] + box_b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union  = area_a + area_b - intersection

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def estimate_gt_mask_from_label(image: np.ndarray, class_id: int) -> np.ndarray:
    """
    Create a simple proxy ground-truth mask when real annotations are not available.

    Strategy: the whole image is assumed to contain the traffic sign (common for
    GTSRB which is already cropped). A full-image foreground mask is used so
    that the IoU score reflects how much of the sign the segmentation captured,
    rather than precision of a tight bounding box.

    For a real evaluation with polygon/bbox annotations, replace this function
    with one that loads your annotation file.

    Returns:
        Binary mask (uint8) the same size as `image`, all 255 (full foreground).
    """
    h, w = image.shape[:2]
    return np.full((h, w), 255, dtype=np.uint8)


def evaluate_segmentation_iou(
    images: List[Tuple[str, np.ndarray]],
    labels: List[int],
    n_samples: int = 200,
) -> dict:
    """
    Estimate mean IoU for the segmentation stage.

    Because GTSRB images are pre-cropped around the sign, the ground-truth
    mask proxy is the full image (see `estimate_gt_mask_from_label`). Replace
    the proxy with real annotations if you have them.

    Args:
        images   : List of (name, bgr_image) tuples.
        labels   : Corresponding integer class ids.
        n_samples: How many images to sample (IoU evaluation is fast, but
                   limiting keeps the overall eval time reasonable).

    Returns:
        dict with keys: mean_iou, std_iou, median_iou, sample_count.
    """
    n = min(n_samples, len(images))
    indices = random.sample(range(len(images)), n)

    iou_scores = []
    for idx in indices:
        name, img = images[idx]
        class_id  = labels[idx]

        _, _, pred_mask = segment_traffic_signs(img)

        gt_mask = estimate_gt_mask_from_label(img, class_id)

        # Resize pred_mask to match gt if needed (segmentation runs on original size)
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        iou = compute_mask_iou(pred_mask, gt_mask)
        iou_scores.append(iou)

    return {
        "mean_iou":    float(np.mean(iou_scores)),
        "std_iou":     float(np.std(iou_scores)),
        "median_iou":  float(np.median(iou_scores)),
        "sample_count": n,
    }


# ─────────────────────────────────────────────
#  DATASET LOADER  (unchanged)
# ─────────────────────────────────────────────

def load_dataset(
    image_dir: str,
    labels_csv: str,
    train_ratio: float = 0.7
) -> Tuple[List, List, List, List, List]:
    """Load GTSRB dataset with class subdirectories."""

    df_labels = pd.read_csv(labels_csv)
    max_class_id = df_labels['ClassId'].max()
    class_names = [''] * (max_class_id + 1)
    for _, row in df_labels.iterrows():
        class_names[row['ClassId']] = row['Name']

    log.info(f"Loaded {len([c for c in class_names if c])} traffic sign classes (0-{max_class_id})")

    image_paths = []
    image_labels = []
    data_path = Path(image_dir)

    for item in data_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            class_id = int(item.name)
            for img_file in item.iterdir():
                if img_file.suffix.lower() in {'.ppm', '.jpg', '.jpeg', '.png', '.bmp'}:
                    image_paths.append(str(img_file))
                    image_labels.append(class_id)

    if not image_paths:
        raise ValueError(f"No class subdirectories found in {image_dir}")

    log.info(f"Found {len(image_paths)} images across {len(set(image_labels))} classes")

    valid_images = []
    valid_labels = []

    for path, label in zip(image_paths, image_labels):
        img = cv2.imread(path)
        if img is not None and img.shape[0] >= 32 and img.shape[1] >= 32:
            valid_images.append((Path(path).stem, img))
            valid_labels.append(label)

    log.info(f"Successfully loaded {len(valid_images)} valid images")

    indices = list(range(len(valid_labels)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=1 - train_ratio,
        stratify=valid_labels,
        random_state=42
    )

    train_images = [valid_images[i] for i in train_idx]
    train_labels = [valid_labels[i] for i in train_idx]
    test_images  = [valid_images[i] for i in test_idx]
    test_labels  = [valid_labels[i] for i in test_idx]

    log.info(f"Train: {len(train_images)} images, {len(set(train_labels))} classes")
    log.info(f"Test:  {len(test_images)} images,  {len(set(test_labels))} classes")

    return train_images, train_labels, test_images, test_labels, class_names


# ─────────────────────────────────────────────
#  MAIN EVALUATION
# ─────────────────────────────────────────────

def evaluate(
    test_images:   List[Tuple[str, np.ndarray]],
    test_labels:   List[int],
    class_names:   List[str],
    train_images:  List[Tuple[str, np.ndarray]],
    train_labels:  List[int],
    output_dir:    str = "eval_output",
) -> dict:
    """Train classifier and evaluate on test images."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Train classifier ──────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("TRAINING CLASSIFIER  (Naive Bayes)")
    log.info("=" * 50)

    unique_classes = set(train_labels)
    if len(unique_classes) < 2:
        log.error(f"Only {len(unique_classes)} class found! Need at least 2 classes.")
        return {}

    log.info(f"Training on {len(train_images)} images from {len(unique_classes)} classes")

    classifier = TrafficSignClassifier()
    train_imgs = [img for _, img in train_images]

    try:
        train_metrics = classifier.train(train_imgs, train_labels)
        log.info(f"Training Accuracy: {train_metrics.get('accuracy', 0):.4f}")
    except Exception as e:
        log.error(f"Training failed: {e}")
        return {}

    # Save trained classifier so it can be reloaded by predict.py
    model_path = os.path.join(output_dir, "classifier.pkl")
    classifier.save(model_path, class_names=class_names)
    log.info(f"Model saved to {model_path}  (use predict.py to test on single images)")

    # ── Segmentation IoU (results printed in final report below) ─────────
    log.info("Running segmentation IoU evaluation...")
    iou_results = evaluate_segmentation_iou(test_images, test_labels, n_samples=200)
    log.info("Segmentation IoU done.")

    # ── Run pipeline on test images ───────────────────────────────────────
    log.info("=" * 50)
    log.info(f"EVALUATING ON {len(test_images)} TEST IMAGES")
    log.info("=" * 50)

    all_results      = []
    true_labels      = []
    pred_labels      = []
    pred_confidences = []

    for idx, (name, img) in enumerate(test_images):
        if idx % 100 == 0 and idx > 0:
            log.info(f"  Progress: {idx}/{len(test_images)}")

        result = run_pipeline(
            img,
            classifier=classifier,
            class_names=class_names,
            output_dir=output_dir,
            image_name=name,
            verbose=False,
        )
        all_results.append(result)

        true_label = test_labels[idx]
        true_labels.append(true_label)

        pred = result["pred_label"]
        pred_confidences.append(result["pred_conf"])

        if pred in class_names:
            pred_labels.append(class_names.index(pred))
        elif isinstance(pred, str) and pred.isdigit():
            pred_labels.append(int(pred))
        else:
            pred_labels.append(-1)

    # ── Classification metrics ────────────────────────────────────────────
    valid_mask  = [p != -1 for p in pred_labels]
    valid_true  = [t for t, m in zip(true_labels, valid_mask) if m]
    valid_pred  = [p for p, m in zip(pred_labels, valid_mask) if m]

    if valid_true:
        metrics = {
            "accuracy":  accuracy_score(valid_true, valid_pred),
            "precision": precision_score(valid_true, valid_pred, average="weighted", zero_division=0),
            "recall":    recall_score(valid_true,    valid_pred, average="weighted", zero_division=0),
            "f1":        f1_score(valid_true,        valid_pred, average="weighted", zero_division=0),
        }

        log.info("=" * 50)
        log.info("TEST RESULTS")
        log.info("=" * 50)
        log.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        log.info(f"  Precision: {metrics['precision']:.4f}")
        log.info(f"  Recall:    {metrics['recall']:.4f}")
        log.info(f"  F1-Score:  {metrics['f1']:.4f}")

        # Confusion matrix
        unique_cls   = sorted(set(valid_true))
        n_classes    = len(unique_cls)
        log.info(f"  Generating confusion matrix for {n_classes} classes...")

        cm = confusion_matrix(valid_true, valid_pred, labels=unique_cls)

        display_labels = []
        for i in unique_cls:
            name = class_names[i] if i < len(class_names) else str(i)
            display_labels.append(name[:18] + ".." if len(name) > 20 else name)

        fig_size = min(25, max(10, n_classes * 0.35))
        fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d')

        ax.set_title(
            f'Confusion Matrix - {n_classes} Traffic Sign Classes\n'
            f'(Accuracy: {metrics["accuracy"]:.2%})',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel('Predicted Class', fontsize=11)
        ax.set_ylabel('True Class',      fontsize=11)

        if n_classes > 30:
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
        elif n_classes > 20:
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

        plt.tight_layout()
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        log.info(f"  Confusion matrix saved to {cm_path}")
        plt.close(fig)

        # Detailed text report
        cm_text_path = os.path.join(output_dir, "confusion_matrix_details.txt")
        with open(cm_text_path, 'w', encoding='utf-8') as f:
            f.write("CONFUSION MATRIX - DETAILED REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Total Test Images: {len(valid_true)}\n\n")
            f.write("Per-Class Performance:\n")
            f.write("-" * 80 + "\n")

            for i, cls in enumerate(unique_cls):
                cname    = class_names[cls] if cls < len(class_names) else str(cls)
                true_pos = cm[i, i]
                false_pos = cm[:, i].sum() - true_pos
                false_neg = cm[i, :].sum() - true_pos
                prec = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                rec  = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1v  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

                f.write(f"\nClass {cls}: {cname}\n")
                f.write(f"  Correct: {true_pos}/{cm[i, :].sum()} "
                        f"({true_pos / cm[i, :].sum() * 100:.1f}%)\n")
                f.write(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1v:.4f}\n")

        log.info(f"  Detailed per-class results saved to {cm_text_path}")

    else:
        metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
        log.warning("No valid predictions to evaluate")

    # ── Timing stats ──────────────────────────────────────────────────────
    stages = ["preprocessing", "harris", "pyramid", "sift",
              "segmentation", "classification", "total"]
    timing = {}
    for stage in stages:
        vals = [r["timing"].get(stage, 0) for r in all_results]
        timing[stage] = {"mean_ms": np.mean(vals) * 1000,
                         "std_ms":  np.std(vals)  * 1000}

    sift_accs     = [r["sift_good"] / r["sift_total"] if r["sift_total"] > 0 else 0
                     for r in all_results]
    corner_counts = [len(r["corners"]) for r in all_results]
    box_counts    = [r["seg_boxes"]    for r in all_results]

    # ── Print final report ────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  FINAL EVALUATION REPORT")
    print("═" * 70)
    print(f"\n  Test Images:        {len(test_images)}")
    print(f"\n  CLASSIFICATION (Naive Bayes):")
    print(f"    Accuracy:         {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"    Precision:        {metrics['precision']:.4f}")
    print(f"    Recall:           {metrics['recall']:.4f}")
    print(f"    F1-Score:         {metrics['f1']:.4f}")
    print(f"\n  SEGMENTATION IoU (n={iou_results['sample_count']}):")
    print(f"    Mean IoU:         {iou_results['mean_iou']:.4f}")
    print(f"    Median IoU:       {iou_results['median_iou']:.4f}")
    print(f"    Std IoU:          {iou_results['std_iou']:.4f}")
    print(f"\n  SIFT MATCHING:")
    print(f"    Avg match rate:   {np.mean(sift_accs):.4f}")
    print(f"\n  HARRIS CORNERS:")
    print(f"    Avg per image:    {np.mean(corner_counts):.1f} ± {np.std(corner_counts):.1f}")
    print(f"\n  SEGMENTATION BOXES:")
    print(f"    Avg boxes/image:  {np.mean(box_counts):.2f}")
    print(f"    Total boxes:      {sum(box_counts)}")
    print(f"\n  TIMING (ms per image):")
    for stage, t in timing.items():
        print(f"    {stage:<15} {t['mean_ms']:7.2f} ± {t['std_ms']:.2f} ms")
    print("═" * 70)

    # ── Save text report ──────────────────────────────────────────────────
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("TRAFFIC SIGN PIPELINE - EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test Images: {len(test_images)}\n\n")
        f.write("CLASSIFICATION METRICS (Naive Bayes):\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics['f1']:.4f}\n\n")
        f.write(f"SEGMENTATION IoU (n={iou_results['sample_count']} samples):\n")
        f.write(f"  Mean IoU:   {iou_results['mean_iou']:.4f}\n")
        f.write(f"  Median IoU: {iou_results['median_iou']:.4f}\n")
        f.write(f"  Std IoU:    {iou_results['std_iou']:.4f}\n\n")
        f.write("TIMING (ms per image):\n")
        for stage, t in timing.items():
            f.write(f"  {stage}: {t['mean_ms']:.2f} ± {t['std_ms']:.2f}\n")

    log.info(f"Report saved to {report_path}")

    return {**metrics, "iou": iou_results}


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",   required=True,
                        help="Directory containing class subfolders (0/, 1/, 2/, ...)")
    parser.add_argument("--labels",      required=True, help="Path to labels.csv")
    parser.add_argument("--output_dir",  default="eval_results", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Train/test split ratio")
    args = parser.parse_args()

    log.info("Loading dataset...")
    train_images, train_labels, test_images, test_labels, class_names = load_dataset(
        args.image_dir, args.labels, args.train_ratio
    )

    evaluate(
        test_images=test_images,
        test_labels=test_labels,
        class_names=class_names,
        train_images=train_images,
        train_labels=train_labels,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
