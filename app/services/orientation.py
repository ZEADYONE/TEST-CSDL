"""Orientation-normalization extension point.

Version 1 deliberately does not infer a bird's head from a silhouette. That
problem is not reliably solvable from the mask alone. A manual query flip is
supported, while this hook keeps future methods isolated from preprocessing.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_orientation(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str = "none",
    flip_horizontal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an orientation-normalized image and mask.

    ``mode='none'`` is the only automatic mode in the current feature version. The
    explicit flip is intended for a user's controlled experiment.
    """

    if mode != "none":
        raise ValueError(f"Unsupported orientation mode: {mode}")
    if flip_horizontal:
        return cv2.flip(image, 1), cv2.flip(mask, 1)
    return image, mask
