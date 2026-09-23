"""Mask-aware fine-grained circular HSV color-spatial descriptor."""

from __future__ import annotations

import cv2
import numpy as np

from app.services.preprocessing import mask_bbox


JOINT_BINS = (18, 4, 4)
SPATIAL_BINS = (12, 3, 2)
JOINT_DIM = int(np.prod(JOINT_BINS))
SPATIAL_REGION_DIM = int(np.prod(SPATIAL_BINS))
SPATIAL_REGIONS = 3
SPATIAL_DIM = SPATIAL_REGIONS * SPATIAL_REGION_DIM
COLOR_DIM = JOINT_DIM + SPATIAL_DIM


def _joint_hsv_histogram(
    hsv: np.ndarray,
    mask: np.ndarray,
    bins: tuple[int, int, int],
    *,
    minimum_pixels: int = 1,
) -> np.ndarray:
    pixels = hsv[mask.astype(bool)]
    dimension = int(np.prod(bins))
    if len(pixels) < minimum_pixels:
        return np.zeros(dimension, dtype=np.float32)
    # Hue is unreliable for nearly gray pixels. Keep a small floor so their
    # value/saturation information remains available without allowing their
    # arbitrary hue to dominate genuinely chromatic plumage.
    saturation_weights = 0.10 + 0.90 * pixels[:, 1].astype(np.float32) / 255.0
    histogram, _ = np.histogramdd(
        pixels.astype(np.float32),
        bins=bins,
        range=((0, 180), (0, 256), (0, 256)),
        weights=saturation_weights,
    )
    histogram = histogram.astype(np.float32)
    # OpenCV hue is circular: red can occur close to both 0 and 179. Circular
    # smoothing makes those boundary bins adjacent while retaining enough
    # resolution to separate red from orange/yellow.
    histogram = (
        0.50 * histogram
        + 0.25 * np.roll(histogram, 1, axis=0)
        + 0.25 * np.roll(histogram, -1, axis=0)
    )
    histogram = histogram.ravel()
    total = float(histogram.sum())
    if total > 0:
        histogram /= total
    return histogram


def extract_color_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract 288D global HSV plus three 72D spatial HSV histograms."""

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask sizes differ")
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        raise ValueError("Cannot extract color from an empty mask")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    global_histogram = _joint_hsv_histogram(hsv, binary, JOINT_BINS)

    bbox = mask_bbox(binary)
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    boundaries = np.linspace(x1, x2, SPATIAL_REGIONS + 1, dtype=int)
    regions: list[np.ndarray] = []
    for index in range(SPATIAL_REGIONS):
        region_mask = np.zeros_like(binary)
        region_mask[y1:y2, boundaries[index] : boundaries[index + 1]] = binary[
            y1:y2, boundaries[index] : boundaries[index + 1]
        ]
        regions.append(
            _joint_hsv_histogram(hsv, region_mask, SPATIAL_BINS, minimum_pixels=8)
        )
    vector = np.concatenate((global_histogram, *regions)).astype(np.float32)
    if vector.shape != (COLOR_DIM,):
        raise AssertionError(f"Expected {COLOR_DIM} color features, got {vector.shape}")
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
