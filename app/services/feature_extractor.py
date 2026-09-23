"""Unified handcrafted feature extraction API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.color_features import COLOR_DIM, extract_color_features
from app.services.gabor_features import GABOR_DIM, extract_gabor_features
from app.services.hog_features import HOG_DIM, extract_hog_features
from app.services.shape_features import SHAPE_DIM as GEOMETRY_DIM, extract_shape_features
from app.services.texture_features import TEXTURE_DIM as BASE_TEXTURE_DIM, extract_texture_features


TEXTURE_DIM = BASE_TEXTURE_DIM + GABOR_DIM
SHAPE_DIM = GEOMETRY_DIM + HOG_DIM


@dataclass(frozen=True)
class FeatureBundle:
    color: np.ndarray
    texture: np.ndarray
    shape: np.ndarray

    def validate(self) -> "FeatureBundle":
        expected = ((self.color, COLOR_DIM), (self.texture, TEXTURE_DIM), (self.shape, SHAPE_DIM))
        for vector, dimension in expected:
            if vector.shape != (dimension,):
                raise AssertionError(f"Expected ({dimension},), got {vector.shape}")
            if not np.all(np.isfinite(vector)):
                raise ValueError("Feature vector contains NaN or infinity")
        return self


class BirdFeatureExtractor:
    """Extract the three independent, interpretable feature groups."""

    def extract(self, image: np.ndarray, mask: np.ndarray) -> FeatureBundle:
        return FeatureBundle(
            color=extract_color_features(image, mask),
            texture=np.concatenate(
                (extract_texture_features(image, mask), extract_gabor_features(image, mask))
            ).astype(np.float32),
            shape=np.concatenate(
                (extract_shape_features(mask), extract_hog_features(image, mask))
            ).astype(np.float32),
        ).validate()
