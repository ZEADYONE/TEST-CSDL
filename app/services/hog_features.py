"""Mask-aware multi-region Histogram of Oriented Gradients descriptor."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog


HOG_BINS = 9
HOG_SIZE = 128
HOG_CELL_SIZE = 16
HOG_BLOCK_CELLS = 2
HOG_CELLS_PER_AXIS = HOG_SIZE // HOG_CELL_SIZE
HOG_BLOCKS_PER_AXIS = HOG_CELLS_PER_AXIS - HOG_BLOCK_CELLS + 1
HOG_REGION_DIM = HOG_BLOCK_CELLS * HOG_BLOCK_CELLS * HOG_BINS
HOG_DIM = HOG_BLOCKS_PER_AXIS * HOG_BLOCKS_PER_AXIS * HOG_REGION_DIM


def extract_hog_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract HOG from 49 overlapping local regions of the normalized bird.

    The masked grayscale image is reduced to 128x128, divided into 16x16
    cells, and described by overlapping 2x2-cell blocks. Each of the 49 local
    blocks receives independent L2-Hys normalization. Background pixels are
    zeroed before resizing, so their original color cannot affect gradients.
    """

    binary = np.asarray(mask, dtype=bool)
    if image.shape[:2] != binary.shape or not np.any(binary):
        raise ValueError("HOG extraction requires a non-empty matching mask")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray[~binary] = 0.0
    resized = cv2.resize(gray, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    vector = hog(
        resized,
        orientations=HOG_BINS,
        pixels_per_cell=(HOG_CELL_SIZE, HOG_CELL_SIZE),
        cells_per_block=(HOG_BLOCK_CELLS, HOG_BLOCK_CELLS),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)
    if vector.shape != (HOG_DIM,):
        raise AssertionError(f"Expected {HOG_DIM} HOG features, got {vector.shape}")
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
