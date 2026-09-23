"""Robust 69-dimensional silhouette and geometric shape descriptor."""

from __future__ import annotations

import cv2
import numpy as np

from app.services.preprocessing import mask_bbox


GRID_SIZE = 8
SILHOUETTE_DIM = GRID_SIZE * GRID_SIZE
SHAPE_SCALAR_DIM = 5
SHAPE_DIM = SILHOUETTE_DIM + SHAPE_SCALAR_DIM


def _silhouette_grid(mask: np.ndarray) -> np.ndarray:
    cells: list[float] = []
    for rows in np.array_split(mask, GRID_SIZE, axis=0):
        for cell in np.array_split(rows, GRID_SIZE, axis=1):
            cells.append(float(np.mean(cell)) if cell.size else 0.0)
    return np.asarray(cells, dtype=np.float32)


def _eccentricity(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return 0.0
    coordinates = np.column_stack((xs, ys)).astype(np.float64)
    covariance = np.cov(coordinates, rowvar=False, bias=True)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    if eigenvalues[0] <= 1e-12:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - eigenvalues[1] / eigenvalues[0])))


def _shape_scalars(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    area = float(binary.sum())
    bbox = mask_bbox(binary)
    if area <= 0 or bbox is None:
        return np.zeros(SHAPE_SCALAR_DIM, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    width, height = float(x2 - x1), float(y2 - y1)
    aspect_ratio = width / height if height > 0 else 0.0
    contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    circularity = float(4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
    all_points = np.vstack(contours) if contours else np.empty((0, 1, 2), dtype=np.int32)
    hull_area = float(cv2.contourArea(cv2.convexHull(all_points))) if len(all_points) >= 3 else 0.0
    solidity = float(np.clip(area / hull_area, 0.0, 1.0)) if hull_area > 0 else 0.0
    bbox_area = width * height
    extent = float(np.clip(area / bbox_area, 0.0, 1.0)) if bbox_area > 0 else 0.0
    return np.asarray(
        [aspect_ratio, _eccentricity(binary), circularity, solidity, extent],
        dtype=np.float32,
    )


def extract_shape_features(mask: np.ndarray) -> np.ndarray:
    """Extract an 8x8 occupancy grid followed by five geometry scalars."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not np.any(binary):
        raise ValueError("Shape extraction requires a non-empty 2D mask")
    vector = np.concatenate((_silhouette_grid(binary), _shape_scalars(binary))).astype(np.float32)
    if vector.shape != (SHAPE_DIM,):
        raise AssertionError(f"Expected {SHAPE_DIM} shape features, got {vector.shape}")
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
