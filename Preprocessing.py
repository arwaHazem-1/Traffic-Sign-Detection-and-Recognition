"""
preprocessing.py — Member 2: Preprocessing Specialist
GTSRB Traffic Sign Recognition Pipeline

Compatibility with Member 1's data_loader.py:
  - Member 1 uses PIL (Pillow) → images are RGB, NOT BGR
  - Member 1 default format is "L" (grayscale) or "RGB"
  - Member 1 default size is (128, 128)
  - Member 1 already handles augmentation (flip, rotation, color jitter)
    via torchvision transforms → our augmented_pipeline() is DISABLED
    to avoid double-augmentation
  - Member 1 output is a PyTorch tensor (C x H x W)
    → our preprocessing runs on PIL/NumPy BEFORE ToTensor()
    → we expose preprocess_pil() to plug into their pipeline

Integration point:
  Member 1 should call preprocess_pil(img) inside their process_img()
  function BEFORE applying torchvision transforms, like so:

      from preprocessing import preprocess_pil
      img = preprocess_pil(img)   # still a PIL Image after this
      transform(img)              # Member 1's existing transform chain

All internal functions work on NumPy arrays (uint8, RGB order).
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


# ─────────────────────────────────────────────
#  CONSTANTS — synced with Member 1
# ─────────────────────────────────────────────

DEFAULT_SIZE: Tuple[int, int] = (128, 128)  # matches Member 1's default size
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # RGB order
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # RGB order


# ─────────────────────────────────────────────
#  PIL <-> NumPy CONVERSION HELPERS
#  Member 1 uses PIL; we work in NumPy/OpenCV internally.
#  All color channels stay in RGB order (NOT BGR).
# ─────────────────────────────────────────────

def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a NumPy uint8 array.

    Handles both:
      - "RGB"  -> (H, W, 3)  uint8
      - "L"    -> (H, W)     uint8  (grayscale, Member 1's default)

    Args:
        img: PIL Image object.

    Returns:
        NumPy uint8 array in RGB order (or 2D for grayscale).
    """
    return np.array(img, dtype=np.uint8)


def numpy_to_pil(img: np.ndarray, mode: Optional[str] = None) -> Image.Image:
    """
    Convert a NumPy uint8 array back to a PIL Image.

    Args:
        img : NumPy uint8 array — (H, W, 3) for RGB or (H, W) for grayscale.
        mode: PIL mode string. Auto-detected if None.

    Returns:
        PIL Image object ready for Member 1's torchvision transforms.
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if mode is None:
        mode = "L" if img.ndim == 2 else "RGB"
    return Image.fromarray(img, mode=mode)


# ─────────────────────────────────────────────
#  1. RESIZE
# ─────────────────────────────────────────────

def resize_image(img: np.ndarray,
                 size: Tuple[int, int] = DEFAULT_SIZE,
                 interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Resize image to (width, height).

    NOTE: Member 1's Resize() in torchvision will also resize,
    so this step is optional. Kept here for standalone use or when
    preprocessing steps need a fixed size (e.g., CLAHE tiling).

    Args:
        img          : Input image (H x W x C) or (H x W), uint8, RGB.
        size         : Target (width, height). Default (128, 128).
        interpolation: cv2.INTER_AREA for downscaling (default).

    Returns:
        Resized image as uint8 NumPy array.
    """
    if img is None or img.size == 0:
        raise ValueError("resize_image: received empty image.")
    return cv2.resize(img, size, interpolation=interpolation)


# ─────────────────────────────────────────────
#  2. NORMALIZATION
#  NOTE: Member 1 uses transforms.ToTensor() which auto-normalizes
#  to [0,1]. These helpers are for use by other members (e.g.,
#  Member 7 — Classification) who need normalized NumPy arrays.
# ─────────────────────────────────────────────

def normalize(img: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize pixel values to float32.

    Args:
        img    : Input image (uint8 or float32), RGB order.
        method : One of:
                   "minmax"   — scales to [0, 1]
                   "zscore"   — zero mean, unit variance
                   "imagenet" — ImageNet mean/std (RGB), expects uint8 input
                   "uint8"    — divides by 255

    Returns:
        Normalized image as float32 NumPy array.
    """
    img = img.astype(np.float32)

    if method == "minmax":
        min_val, max_val = img.min(), img.max()
        if max_val - min_val == 0:
            return np.zeros_like(img)
        return (img - min_val) / (max_val - min_val)

    elif method == "zscore":
        mean, std = img.mean(), img.std()
        if std == 0:
            return np.zeros_like(img)
        return (img - mean) / std

    elif method == "imagenet":
        # Expects RGB uint8 input, shape (H, W, 3)
        img /= 255.0
        return (img - IMAGENET_MEAN) / IMAGENET_STD

    elif method == "uint8":
        return img / 255.0

    else:
        raise ValueError(f"normalize: unknown method '{method}'. "
                         f"Choose from: minmax, zscore, imagenet, uint8.")


# ─────────────────────────────────────────────
#  3. GRAYSCALE CONVERSION
# ─────────────────────────────────────────────

def to_grayscale(img: np.ndarray, keep_dims: bool = False) -> np.ndarray:
    """
    Convert an RGB image to grayscale.

    Args:
        img       : RGB image (H x W x 3) or already grayscale (H x W).
        keep_dims : If True, return shape (H x W x 1); else (H x W).

    Returns:
        Grayscale image as uint8.
    """
    if img.ndim == 2:
        gray = img  # already grayscale
    else:
        # Member 1 uses RGB (PIL), so use RGB2GRAY (not BGR2GRAY)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if keep_dims:
        return gray[:, :, np.newaxis]
    return gray


# ─────────────────────────────────────────────
#  4. COLOR SPACE CONVERSIONS
#  All inputs are RGB (from PIL via Member 1).
# ─────────────────────────────────────────────

def to_hsv(img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to HSV color space.

    HSV separates luminance (V) from color (H, S), making
    color-based segmentation more robust to lighting changes.
    Used by Member 6 (Segmentation Engineer) for color masking.

    Args:
        img: RGB image (H x W x 3), uint8.

    Returns:
        HSV image (H x W x 3), uint8.
        H in [0,179], S in [0,255], V in [0,255].
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("to_hsv: input must be a 3-channel RGB image.")
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def to_lab(img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to CIE L*a*b* color space.

    L*a*b* is perceptually uniform — used internally for CLAHE
    equalization on the L channel only.

    Args:
        img: RGB image (H x W x 3), uint8.

    Returns:
        L*a*b* image (H x W x 3), uint8.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("to_lab: input must be a 3-channel RGB image.")
    return cv2.cvtColor(img, cv2.COLOR_RGB2Lab)


# ─────────────────────────────────────────────
#  5. HISTOGRAM EQUALIZATION
# ─────────────────────────────────────────────

def histogram_equalization(img: np.ndarray,
                            method: str = "clahe",
                            clip_limit: float = 2.0,
                            tile_grid_size: Tuple[int, int] = (8, 8)
                            ) -> np.ndarray:
    """
    Enhance contrast using histogram equalization.

    "clahe"  — Contrast Limited Adaptive Histogram Equalization (recommended).
               Applied to the L channel of L*a*b* for color images,
               or directly on grayscale.

    "global" — Standard OpenCV equalizeHist. Fast but may over-amplify noise.
               Applied to V channel of HSV for color images.

    Mathematical basis (CDF equalization):
        s_k = (L - 1) * CDF(r_k)
        where CDF(r_k) = sum of p(r_j) for j=0..k

    Args:
        img           : RGB or grayscale uint8 image.
        method        : "clahe" (default) or "global".
        clip_limit    : CLAHE clip limit (higher = more contrast, more noise).
        tile_grid_size: CLAHE tile grid size.

    Returns:
        Contrast-enhanced image, same shape and dtype as input.
    """
    is_color = (img.ndim == 3 and img.shape[2] == 3)

    if method == "global":
        if is_color:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:
            return cv2.equalizeHist(img)

    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                 tileGridSize=tile_grid_size)
        if is_color:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        else:
            return clahe.apply(img)

    else:
        raise ValueError(f"histogram_equalization: unknown method '{method}'. "
                         f"Choose 'clahe' or 'global'.")


# ─────────────────────────────────────────────
#  6. GAUSSIAN BLUR
# ─────────────────────────────────────────────

def apply_gaussian_blur(img: np.ndarray,
                        kernel_size: Tuple[int, int] = (3, 3),
                        sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian blur to smooth the image and reduce high-frequency noise.

    Mathematical basis:
        G(x, y) = (1 / 2*pi*sigma^2) * exp(-(x^2 + y^2) / 2*sigma^2)
        Output = Image convolved with G

    Args:
        img        : Input image (any channels), uint8.
        kernel_size: (width, height) — both must be odd positive integers.
        sigma      : Gaussian std dev. If 0, OpenCV computes from kernel size.

    Returns:
        Blurred image, same shape as input.
    """
    kw, kh = kernel_size
    if kw % 2 == 0 or kh % 2 == 0:
        raise ValueError("apply_gaussian_blur: kernel dimensions must be odd.")
    return cv2.GaussianBlur(img, kernel_size, sigma)


# ─────────────────────────────────────────────
#  7. NOISE REMOVAL
# ─────────────────────────────────────────────

def remove_noise(img: np.ndarray,
                 method: str = "nlm",
                 median_ksize: int = 3,
                 nlm_h: float = 10.0,
                 nlm_template_size: int = 7,
                 nlm_search_size: int = 21) -> np.ndarray:
    """
    Remove noise from the image.

    "nlm"    — Non-Local Means: compares patches across the image,
               averages similar ones. Best quality for low-light signs.

    "median" — Replaces each pixel with the neighborhood median.
               Very effective against salt-and-pepper noise. Faster.

    Args:
        img              : Input image (RGB or grayscale), uint8.
        method           : "nlm" (default) or "median".
        median_ksize     : Kernel size for median filter (must be odd).
        nlm_h            : NLM filter strength (5-15 typical range).
        nlm_template_size: NLM patch size (must be odd).
        nlm_search_size  : NLM search window size (must be odd).

    Returns:
        Denoised image, same shape as input.
    """
    if method == "median":
        if median_ksize % 2 == 0:
            raise ValueError("remove_noise: median_ksize must be odd.")
        return cv2.medianBlur(img, median_ksize)

    elif method == "nlm":
        if img.ndim == 2:
            return cv2.fastNlMeansDenoising(
                img,
                h=nlm_h,
                templateWindowSize=nlm_template_size,
                searchWindowSize=nlm_search_size
            )
        else:
            # fastNlMeansDenoisingColored expects BGR internally,
            # so convert RGB -> BGR -> denoise -> BGR -> RGB
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            denoised_bgr = cv2.fastNlMeansDenoisingColored(
                bgr,
                h=nlm_h,
                hColor=nlm_h,
                templateWindowSize=nlm_template_size,
                searchWindowSize=nlm_search_size
            )
            return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)

    else:
        raise ValueError(f"remove_noise: unknown method '{method}'. "
                         f"Choose 'nlm' or 'median'.")


# ─────────────────────────────────────────────
#  8. AUGMENTATION HELPERS
#  NOTE: Member 1 already handles augmentation via torchvision:
#    - RandomHorizontalFlip
#    - RandomRotation(15)
#    - ColorJitter(brightness=0.2, contrast=0.2)
#
#  These helpers are kept for standalone/reference use only.
#  Do NOT call these when using Member 1's loader (double augmentation).
# ─────────────────────────────────────────────

def random_rotation(img: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Random rotation +/-max_angle degrees. Already done by Member 1."""
    angle = np.random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def random_brightness(img: np.ndarray, delta: float = 30.0) -> np.ndarray:
    """Random brightness shift +/-delta. Already done by Member 1 (ColorJitter)."""
    shift = np.random.uniform(-delta, delta)
    return np.clip(img.astype(np.float32) + shift, 0, 255).astype(np.uint8)


def horizontal_flip(img: np.ndarray) -> np.ndarray:
    """Horizontal flip. Already done by Member 1 (RandomHorizontalFlip)."""
    return cv2.flip(img, 1)


def random_zoom(img: np.ndarray,
                zoom_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Random zoom in/out, crop/pad back to original size."""
    h, w = img.shape[:2]
    zoom = np.random.uniform(*zoom_range)
    new_h, new_w = int(h * zoom), int(w * zoom)
    resized = cv2.resize(img, (new_w, new_h))
    if zoom > 1.0:
        sy, sx = (new_h - h) // 2, (new_w - w) // 2
        return resized[sy:sy + h, sx:sx + w]
    else:
        py, px = (h - new_h) // 2, (w - new_w) // 2
        return cv2.copyMakeBorder(resized,
                                  py, h - new_h - py,
                                  px, w - new_w - px,
                                  cv2.BORDER_REFLECT)


# ─────────────────────────────────────────────
#  9. MAIN INTEGRATION POINT WITH MEMBER 1
# ─────────────────────────────────────────────

def preprocess_pil(pil_img: Image.Image,
                   equalize: bool = True,
                   denoise: bool = True,
                   blur: bool = False) -> Image.Image:
    """
    PRIMARY INTEGRATION FUNCTION FOR MEMBER 1

    Accepts a PIL Image (as Member 1's loader provides),
    applies OpenCV-based preprocessing, and returns a PIL Image
    so Member 1's torchvision transform chain can continue normally.

    Member 1 inserts ONE line in process_img(), before transform(img):

        from preprocessing import preprocess_pil
        img = preprocess_pil(img)   <-- insert this line
        return transform(img)       <-- Member 1's existing line

    Pipeline:
        1. PIL -> NumPy (preserve RGB/L mode)
        2. Denoise (NLM)            optional, default ON
        3. CLAHE contrast enhance   optional, default ON
        4. Gaussian blur            optional, default OFF
        5. NumPy -> PIL (restore mode)

    Resize and augmentation are intentionally left to Member 1's
    torchvision transforms to avoid double processing.

    Args:
        pil_img : PIL Image in "RGB" or "L" mode (from Member 1's loader).
        equalize: Apply CLAHE histogram equalization (default True).
        denoise : Apply NLM denoising (default True).
        blur    : Apply Gaussian blur (default False).

    Returns:
        Preprocessed PIL Image in the same mode as input.
    """
    original_mode = pil_img.mode  # "RGB" or "L"
    img = pil_to_numpy(pil_img)   # -> (H, W, 3) or (H, W), uint8

    if denoise:
        img = remove_noise(img, method="nlm", nlm_h=8.0)

    if equalize:
        img = histogram_equalization(img, method="clahe")

    if blur:
        img = apply_gaussian_blur(img, kernel_size=(3, 3))

    return numpy_to_pil(img, mode=original_mode)


# ─────────────────────────────────────────────
#  10. STANDALONE PIPELINE
#  For use WITHOUT Member 1's loader (e.g., single-image inference,
#  or by Member 3/6 who need preprocessed NumPy arrays directly).
# ─────────────────────────────────────────────

def full_pipeline(img: np.ndarray,
                  size: Tuple[int, int] = DEFAULT_SIZE,
                  normalize_method: str = "minmax",
                  equalize: bool = True,
                  denoise: bool = True,
                  blur: bool = False,
                  output_grayscale: bool = False) -> np.ndarray:
    """
    Standalone preprocessing pipeline — returns a float32 NumPy array.

    Use this when NOT going through Member 1's loader, e.g.:
      - Single image inference
      - Member 6 (Segmentation) needs a preprocessed NumPy array
      - Member 3 (Harris) needs a preprocessed grayscale array

    Input must be RGB uint8 NumPy array.

    Pipeline:
        1. Resize     -> DEFAULT_SIZE (128, 128)
        2. Denoise    -> NLM          (optional)
        3. Equalize   -> CLAHE        (optional)
        4. Blur       -> Gaussian     (optional)
        5. Grayscale                  (optional)
        6. Normalize  -> float32

    Returns:
        float32 NumPy array, shape (H, W) or (H, W, 3).
    """
    img = resize_image(img, size=size, interpolation=cv2.INTER_AREA)

    if denoise:
        img = remove_noise(img, method="nlm", nlm_h=8.0)

    if equalize:
        img = histogram_equalization(img, method="clahe")

    if blur:
        img = apply_gaussian_blur(img, kernel_size=(3, 3))

    if output_grayscale:
        img = to_grayscale(img)

    img = normalize(img, method=normalize_method)
    return img


def preprocess_batch(images: list,
                     size: Tuple[int, int] = DEFAULT_SIZE,
                     normalize_method: str = "minmax") -> np.ndarray:
    """
    Preprocess a list of RGB uint8 NumPy arrays into a stacked batch.

    NOTE: No augmentation here — augmentation is Member 1's responsibility.

    Args:
        images          : List of RGB uint8 NumPy arrays.
        size            : Target image size.
        normalize_method: Normalization method.

    Returns:
        float32 array of shape (N, H, W, C) or (N, H, W).
    """
    processed = [full_pipeline(img, size=size, normalize_method=normalize_method)
                 for img in images]
    return np.stack(processed, axis=0)


# ─────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  preprocessing.py — Member 2 Self-Test")
    print("  (Compatible with Member 1 data_loader.py)")
    print("=" * 60)

    # Simulate Member 1's PIL output (RGB mode)
    arr = np.random.randint(0, 256, (60, 60, 3), dtype=np.uint8)
    arr[10:50, 10:50] = [200, 0, 0]  # RGB red patch (stop sign simulation)
    pil_rgb  = Image.fromarray(arr, "RGB")
    pil_gray = pil_rgb.convert("L")

    print(f"\n[Input PIL RGB]   size={pil_rgb.size}, mode={pil_rgb.mode}")
    print(f"[Input PIL Gray]  size={pil_gray.size}, mode={pil_gray.mode}")

    # Test preprocess_pil (main integration point with Member 1)
    out_rgb  = preprocess_pil(pil_rgb)
    out_gray = preprocess_pil(pil_gray)
    print(f"\n[preprocess_pil RGB]  mode={out_rgb.mode},  size={out_rgb.size}")
    print(f"[preprocess_pil Gray] mode={out_gray.mode}, size={out_gray.size}")

    # Test individual NumPy functions
    img_np = pil_to_numpy(pil_rgb)
    print(f"\n[pil_to_numpy]    shape={img_np.shape}, dtype={img_np.dtype}")

    print(f"[resize_image]    shape={resize_image(img_np).shape}")
    print(f"[to_grayscale]    shape={to_grayscale(img_np).shape}")
    print(f"[to_hsv]          shape={to_hsv(img_np).shape}")
    print(f"[CLAHE equalize]  shape={histogram_equalization(img_np).shape}")
    print(f"[gaussian_blur]   shape={apply_gaussian_blur(img_np).shape}")
    print(f"[denoise median]  shape={remove_noise(img_np, method='median').shape}")

    normed = normalize(img_np, method="minmax")
    print(f"[normalize]       shape={normed.shape}, "
          f"min={normed.min():.3f}, max={normed.max():.3f}")

    result = full_pipeline(img_np)
    print(f"\n[full_pipeline]   shape={result.shape}, dtype={result.dtype}, "
          f"min={result.min():.3f}, max={result.max():.3f}")

    batch = preprocess_batch([img_np, img_np, img_np])
    print(f"[batch N=3]       shape={batch.shape}")

    print("\nAll tests passed.")
    print("=" * 60)
    print("\nIntegration note for Member 1 (data_loader.py):")
    print("  Add inside process_img(), before 'return transform(img)':")
    print()
    print("      from preprocessing import preprocess_pil")
    print("      img = preprocess_pil(img)   # <-- insert this line")
    print("      return transform(img)        # <-- existing line")
    print()
    print("  - preprocess_pil() accepts PIL 'RGB' or 'L' — matches Member 1")
    print("  - Resize/augmentation left to Member 1's torchvision transforms")
    print("  - full_pipeline() for standalone NumPy use (Members 3, 6)")
    print("  - All color ops use RGB order (NOT BGR) — matches PIL/Member 1")