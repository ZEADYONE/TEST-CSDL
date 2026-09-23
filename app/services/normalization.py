"""Dataset-fitted normalization for numerical (non-histogram) features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from app.services.feature_extractor import FeatureBundle
from app.services.shape_features import SHAPE_SCALAR_DIM, SILHOUETTE_DIM
from app.services.texture_features import LBP_DIM, TEXTURE_DIM as BASE_TEXTURE_DIM


@dataclass
class FeatureNormalizers:
    glcm: StandardScaler
    gabor: StandardScaler
    shape_scalars: StandardScaler

    @classmethod
    def fit(cls, bundles: list[FeatureBundle]) -> "FeatureNormalizers":
        if not bundles:
            raise ValueError("Cannot fit scalers without dataset features")
        scalar_end = SILHOUETTE_DIM + SHAPE_SCALAR_DIM
        glcm = StandardScaler().fit(
            np.stack([bundle.texture[LBP_DIM:BASE_TEXTURE_DIM] for bundle in bundles])
        )
        gabor = StandardScaler().fit(
            np.stack([bundle.texture[BASE_TEXTURE_DIM:] for bundle in bundles])
        )
        shape = StandardScaler().fit(
            np.stack([bundle.shape[SILHOUETTE_DIM:scalar_end] for bundle in bundles])
        )
        return cls(glcm=glcm, gabor=gabor, shape_scalars=shape)

    def transform(self, bundle: FeatureBundle) -> FeatureBundle:
        texture = bundle.texture.copy()
        shape = bundle.shape.copy()
        scalar_end = SILHOUETTE_DIM + SHAPE_SCALAR_DIM
        texture[LBP_DIM:BASE_TEXTURE_DIM] = self.glcm.transform(
            texture[LBP_DIM:BASE_TEXTURE_DIM].reshape(1, -1)
        )[0]
        texture[BASE_TEXTURE_DIM:] = self.gabor.transform(
            texture[BASE_TEXTURE_DIM:].reshape(1, -1)
        )[0]
        shape[SILHOUETTE_DIM:scalar_end] = self.shape_scalars.transform(
            shape[SILHOUETTE_DIM:scalar_end].reshape(1, -1)
        )[0]
        return FeatureBundle(bundle.color.copy(), texture.astype(np.float32), shape.astype(np.float32)).validate()

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.glcm, directory / "glcm_scaler.joblib")
        joblib.dump(self.gabor, directory / "gabor_scaler.joblib")
        joblib.dump(self.shape_scalars, directory / "shape_scalar_scaler.joblib")

    @classmethod
    def load(cls, directory: Path) -> "FeatureNormalizers":
        glcm_path = directory / "glcm_scaler.joblib"
        gabor_path = directory / "gabor_scaler.joblib"
        shape_path = directory / "shape_scalar_scaler.joblib"
        if not glcm_path.is_file() or not gabor_path.is_file() or not shape_path.is_file():
            raise FileNotFoundError("Feature scalers are missing. Run scripts/index_dataset.py first.")
        return cls(joblib.load(glcm_path), joblib.load(gabor_path), joblib.load(shape_path))
