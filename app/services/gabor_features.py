"""Mask-aware multi-scale, multi-orientation Gabor texture descriptor."""

from __future__ import annotations

import cv2
import numpy as np


GABOR_SIZE = 128
GABOR_WAVELENGTHS = (4.0, 8.0, 16.0)
GABOR_ORIENTATIONS = tuple(index * np.pi / 6.0 for index in range(6))
GABOR_STATS_PER_FILTER = 2
GABOR_DIM = len(GABOR_WAVELENGTHS) * len(GABOR_ORIENTATIONS) * GABOR_STATS_PER_FILTER
EPSILON = 1e-8


def _kernel(wavelength: float, orientation: float) -> np.ndarray:
    sigma = 0.56 * wavelength
    size = int(np.ceil(6.0 * sigma))
    if size % 2 == 0:
        size += 1
    kernel = cv2.getGaborKernel(
        (size, size),
        sigma,
        orientation,
        wavelength,
        gamma=0.5,
        psi=0.0,
        ktype=cv2.CV_32F,
    )
    kernel -= float(kernel.mean())
    kernel /= float(np.sqrt(np.sum(kernel * kernel)) + EPSILON)
    return kernel


GABOR_KERNELS = tuple(
    (wavelength, _kernel(wavelength, orientation))
    for wavelength in GABOR_WAVELENGTHS
    for orientation in GABOR_ORIENTATIONS
)


def extract_gabor_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return mean and deviation of 18 foreground Gabor magnitude responses."""

    binary = np.asarray(mask, dtype=bool)
    if image.shape[:2] != binary.shape or not np.any(binary):
        raise ValueError("Gabor extraction requires a non-empty matching mask")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    foreground_mean = float(gray[binary].mean())
    gray[~binary] = foreground_mean
    resized = cv2.resize(gray, (GABOR_SIZE, GABOR_SIZE), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(
        binary.astype(np.uint8), (GABOR_SIZE, GABOR_SIZE), interpolation=cv2.INTER_NEAREST
    )

    statistics: list[float] = []
    for wavelength, kernel in GABOR_KERNELS:
        erosion_radius = max(1, int(round(wavelength / 2.0)))
        support = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erosion_radius * 2 + 1, erosion_radius * 2 + 1),
        )
        valid = cv2.erode(resized_mask, support, iterations=1).astype(bool)
        if not np.any(valid):
            valid = resized_mask.astype(bool)
        response = np.abs(cv2.filter2D(resized, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT))
        values = response[valid]
        statistics.extend((float(values.mean()), float(values.std())))

    vector = np.asarray(statistics, dtype=np.float32)
    if vector.shape != (GABOR_DIM,):
        raise AssertionError(f"Expected {GABOR_DIM} Gabor features, got {vector.shape}")
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
