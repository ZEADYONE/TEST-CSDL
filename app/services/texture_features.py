"""Mask-aware 38-dimensional LBP and GLCM texture descriptor."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


LBP_DIM = 28
GLCM_DIM = 10
TEXTURE_DIM = LBP_DIM + GLCM_DIM
GRAY_LEVELS = 32


def _lbp_histogram(gray: np.ndarray, mask: np.ndarray, points: int, radius: int) -> np.ndarray:
    # LBP samples a neighborhood around every center. Erode by the sampling
    # radius so even contour centers cannot read transparent/background pixels.
    # scikit-image bilinearly interpolates circular samples, so keep one extra
    # pixel beyond R and use a square support to cover diagonal interpolation.
    support = radius + 1
    kernel_size = support * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    valid_centers = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    if not np.any(valid_centers):
        valid_centers = mask
    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    values = lbp[valid_centers]
    bins = points + 2
    histogram, _ = np.histogram(values, bins=np.arange(bins + 1), range=(0, bins))
    histogram = histogram.astype(np.float32)
    total = float(histogram.sum())
    if total > 0:
        histogram /= total
    return histogram


def _masked_glcm(
    quantized: np.ndarray,
    mask: np.ndarray,
    dy: int,
    dx: int,
    levels: int = GRAY_LEVELS,
) -> np.ndarray:
    """Build a symmetric GLCM using only pairs whose two pixels are foreground."""

    height, width = mask.shape
    y0_start, y0_stop = max(0, -dy), min(height, height - dy)
    x0_start, x0_stop = max(0, -dx), min(width, width - dx)
    source = quantized[y0_start:y0_stop, x0_start:x0_stop]
    target = quantized[y0_start + dy : y0_stop + dy, x0_start + dx : x0_stop + dx]
    valid = mask[y0_start:y0_stop, x0_start:x0_stop] & mask[
        y0_start + dy : y0_stop + dy, x0_start + dx : x0_stop + dx
    ]
    matrix = np.zeros((levels, levels), dtype=np.float64)
    if np.any(valid):
        np.add.at(matrix, (source[valid], target[valid]), 1.0)
        np.add.at(matrix, (target[valid], source[valid]), 1.0)
    total = float(matrix.sum())
    if total > 0:
        matrix /= total
    return matrix


def _glcm_properties(matrix: np.ndarray) -> tuple[float, float, float, float, float]:
    if not np.any(matrix):
        return 0.0, 0.0, 0.0, 0.0, 0.0
    levels = matrix.shape[0]
    i, j = np.indices((levels, levels), dtype=np.float64)
    delta = i - j
    contrast = float(np.sum(matrix * delta**2))
    energy = float(np.sqrt(np.sum(matrix**2)))
    homogeneity = float(np.sum(matrix / (1.0 + delta**2)))
    marginal_i = matrix.sum(axis=1)
    marginal_j = matrix.sum(axis=0)
    mean_i = float(np.sum(np.arange(levels) * marginal_i))
    mean_j = float(np.sum(np.arange(levels) * marginal_j))
    std_i = float(np.sqrt(np.sum((np.arange(levels) - mean_i) ** 2 * marginal_i)))
    std_j = float(np.sqrt(np.sum((np.arange(levels) - mean_j) ** 2 * marginal_j)))
    denominator = std_i * std_j
    correlation = (
        float(np.sum((i - mean_i) * (j - mean_j) * matrix) / denominator)
        if denominator > 1e-12
        else 1.0
    )
    positive = matrix[matrix > 0]
    entropy = float(-np.sum(positive * np.log2(positive + 1e-12)))
    return contrast, energy, homogeneity, correlation, entropy


def extract_texture_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract 28D multi-scale uniform LBP and 10D masked GLCM statistics."""

    binary = np.asarray(mask, dtype=bool)
    if image.shape[:2] != binary.shape or not np.any(binary):
        raise ValueError("Texture extraction requires a non-empty matching mask")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = np.concatenate(
        (
            _lbp_histogram(gray, binary, points=8, radius=1),
            _lbp_histogram(gray, binary, points=16, radius=2),
        )
    )
    quantized = np.minimum(gray.astype(np.uint16) * GRAY_LEVELS // 256, GRAY_LEVELS - 1).astype(
        np.uint8
    )
    offsets = ((0, 1), (-1, 1), (-1, 0), (-1, -1))
    per_angle = np.asarray(
        [_glcm_properties(_masked_glcm(quantized, binary, dy, dx)) for dy, dx in offsets],
        dtype=np.float64,
    )
    glcm = np.column_stack((per_angle.mean(axis=0), per_angle.std(axis=0))).ravel()
    vector = np.concatenate((lbp, glcm)).astype(np.float32)
    if vector.shape != (TEXTURE_DIM,):
        raise AssertionError(f"Expected {TEXTURE_DIM} texture features, got {vector.shape}")
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
