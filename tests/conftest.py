from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.fixture
def bird_sample() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic colored/textured silhouette with no background contribution."""

    image = np.zeros((384, 384, 3), dtype=np.uint8)
    mask = np.zeros((384, 384), dtype=np.uint8)
    cv2.ellipse(mask, (192, 192), (130, 72), -8, 0, 360, 1, -1)
    cv2.circle(mask, (305, 163), 38, 1, -1)
    triangle = np.asarray([[340, 160], [378, 174], [340, 184]], dtype=np.int32)
    cv2.fillPoly(mask, [triangle], 1)
    yy, xx = np.indices(mask.shape)
    image[:, :, 0] = (xx * 3 + yy) % 256
    image[:, :, 1] = (yy * 2) % 256
    image[:, :, 2] = ((xx // 8 % 2) * 150 + 70).astype(np.uint8)
    image[mask == 0] = (240, 10, 240)  # Deliberately vivid background; must be ignored.
    return image, mask.astype(bool)
