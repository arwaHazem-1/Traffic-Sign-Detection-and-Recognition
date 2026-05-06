import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

GrayImage = np.ndarray
ColorImage = np.ndarray
ResponseMap = np.ndarray
CornerArray = np.ndarray


def compute_gradients(
    gray: GrayImage,
) -> Tuple[GrayImage, GrayImage]:

    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    Ix = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return Ix, Iy


def compute_structure_tensor(
    Ix: GrayImage,
    Iy: GrayImage,
) -> Tuple[GrayImage, GrayImage, GrayImage]:

    kw = max(3, int(6 * 2.0) | 1)
    kernel = cv2.getGaussianKernel(kw, 2.0)
    kernel = kernel @ kernel.T

    Sxx = cv2.filter2D(Ix * Ix, -1, kernel)
    Syy = cv2.filter2D(Iy * Iy, -1, kernel)
    Sxy = cv2.filter2D(Ix * Iy, -1, kernel)
    return Sxx.astype(np.float32), Syy.astype(np.float32), Sxy.astype(np.float32)


def compute_harris_response(
    Sxx: GrayImage,
    Syy: GrayImage,
    Sxy: GrayImage,
) -> ResponseMap:

    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    R = det - 0.04 * trace * trace
    return R.astype(np.float32)


def non_maximum_suppression(
    R: ResponseMap,
) -> ResponseMap:

    window_size = 5
    threshold = 0.01

    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    R_padded = np.pad(R, pad, mode='constant')
    R_dilated = np.zeros_like(R)

    for i in range(pad, R_padded.shape[0] - pad):
        for j in range(pad, R_padded.shape[1] - pad):
            window = R_padded[i-pad:i+pad+1, j-pad:j+pad+1]
            R_dilated[i-pad, j-pad] = window.max()

    local_max = (R == R_dilated)

    abs_thresh = threshold * float(R.max()) if R.max() > 0 else 0.0
    strong = (R > abs_thresh)

    R_nms = R * local_max * strong
    return R_nms.astype(np.float32)


def detect_harris_corners(
    image: ColorImage,
) -> Tuple[CornerArray, ResponseMap]:

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    Ix, Iy = compute_gradients(gray)
    Sxx, Syy, Sxy = compute_structure_tensor(Ix, Iy)
    R = compute_harris_response(Sxx, Syy, Sxy)
    R_nms = non_maximum_suppression(R)

    rows, cols = np.where(R_nms > 0)
    corners = np.stack([rows, cols], axis=1).astype(np.int32)

    return corners, R_nms


def draw_corners(
    image: ColorImage,
    corners: CornerArray,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 4,
    thickness: int = 2,
) -> ColorImage:

    vis = image.copy()
    for row, col in corners:
        cv2.circle(vis, (int(col), int(row)), radius, color, thickness)
    return vis


def visualize_pipeline(
    image: ColorImage,
    corners: CornerArray,
    R_nms: ResponseMap,
    Ix: Optional[GrayImage] = None,
    Iy: Optional[GrayImage] = None,
    title: str = "Harris Corner Detection",
    save_path: Optional[str] = None,
) -> None:

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis_img = draw_corners(image, corners)
    vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    n_panels = 4 if (Ix is not None and Iy is not None) else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    fig.suptitle(f"{title}  |  {len(corners)} corners detected", fontsize=14, fontweight="bold")

    axes[0].imshow(rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    panel = 1
    if Ix is not None and Iy is not None:
        grad_mag = np.sqrt(Ix ** 2 + Iy ** 2)
        axes[panel].imshow(grad_mag, cmap="gray")
        axes[panel].set_title("Gradient Magnitude |∇I|")
        axes[panel].axis("off")
        panel += 1

    axes[panel].imshow(R_nms, cmap="hot")
    axes[panel].set_title("Harris Response R (after NMS)")
    axes[panel].axis("off")

    axes[panel + 1].imshow(vis_rgb)
    axes[panel + 1].set_title(f"Detected Corners (n={len(corners)})")
    axes[panel + 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[harris] Figure saved → {save_path}")
    else:
        plt.show(block=False)

    plt.close(fig)




def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="harris.py",
        description="Harris Corner Detection for traffic sign images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--image", required=True, help="Path to input image.")
    p.add_argument("--output", default=None, help="Save result figure to this path.")

    return p


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)

    image_path = Path(args.image)

    if not image_path.exists():
        sys.exit(f"[harris] ERROR: File not found → {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        sys.exit(f"[harris] ERROR: cv2.imread failed for → {image_path}")

    print(f"[harris] Image loaded: {image_path.name}  {image.shape[1]}×{image.shape[0]}px")

    corners, R_nms = detect_harris_corners(image)

    print(f"[harris] Corners detected: {len(corners)}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Ix, Iy = compute_gradients(gray)

    visualize_pipeline(
        image,
        corners,
        R_nms,
        Ix=Ix,
        Iy=Iy,
        title=image_path.stem,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()