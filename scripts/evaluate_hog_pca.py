"""Evaluate PCA compression of HOG without changing the retrieval objective."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import joinedload

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import ImageFeature, ImageRecord  # noqa: E402
from app.services.color_features import (  # noqa: E402
    JOINT_DIM,
    SPATIAL_REGION_DIM,
    SPATIAL_REGIONS,
)
from app.services.shape_features import (  # noqa: E402
    SHAPE_DIM as GEOMETRY_DIM,
    SILHOUETTE_DIM,
)
from app.services.texture_features import (  # noqa: E402
    LBP_DIM,
    TEXTURE_DIM as BASE_TEXTURE_DIM,
)
from scripts.tune_weights import (  # noqa: E402
    evaluate,
    pairwise_chi_square,
    robust_normalize,
    split_queries,
)


PCA_DIMENSIONS = (32, 64, 128, 256)


def normalized_group(raw: np.ndarray) -> np.ndarray:
    return robust_normalize(raw)[0]


def fuse(
    color: np.ndarray,
    texture: np.ndarray,
    shape: np.ndarray,
    settings,
) -> np.ndarray:
    return (
        settings.weight_color * color
        + settings.weight_texture * texture
        + settings.weight_shape * shape
    )


def main() -> int:
    settings = get_settings()
    with SessionLocal() as session:
        records = list(
            session.scalars(
                select(ImageRecord)
                .options(joinedload(ImageRecord.feature))
                .join(ImageFeature)
                .where(
                    ImageRecord.species.is_not(None),
                    ImageFeature.feature_version == settings.feature_version,
                )
                .order_by(ImageRecord.id)
            ).all()
        )
    labels = np.asarray([record.species for record in records], dtype=object)
    colors = np.asarray([record.feature.color_vector for record in records], dtype=np.float32)
    textures = np.asarray([record.feature.texture_vector for record in records], dtype=np.float32)
    shapes = np.asarray([record.feature.shape_vector for record in records], dtype=np.float32)
    calibration, test = split_queries(labels, settings.random_seed)

    print(
        f"Feature version={settings.feature_version}, records={len(records)}, "
        f"calibration={len(calibration)}, test={len(test)}"
    )
    print("Computing fixed Color, Texture, and geometry distances...")
    joint = pairwise_chi_square(colors[:, :JOINT_DIM])
    spatial = sum(
        pairwise_chi_square(
            colors[
                :,
                JOINT_DIM
                + region * SPATIAL_REGION_DIM : JOINT_DIM
                + (region + 1) * SPATIAL_REGION_DIM,
            ]
        )
        for region in range(SPATIAL_REGIONS)
    ) / SPATIAL_REGIONS
    color = normalized_group(
        settings.color_joint_weight * joint + settings.color_spatial_weight * spatial
    )

    lbp = pairwise_chi_square(textures[:, :LBP_DIM])
    glcm = cdist(
        textures[:, LBP_DIM:BASE_TEXTURE_DIM],
        textures[:, LBP_DIM:BASE_TEXTURE_DIM],
        metric="euclidean",
    ).astype(np.float32)
    gabor = cdist(
        textures[:, BASE_TEXTURE_DIM:], textures[:, BASE_TEXTURE_DIM:], metric="euclidean"
    ).astype(np.float32)
    texture = normalized_group(
        settings.texture_lbp_weight * lbp
        + settings.texture_glcm_weight * glcm
        + settings.texture_gabor_weight * gabor
    )

    silhouette = cdist(
        shapes[:, :SILHOUETTE_DIM], shapes[:, :SILHOUETTE_DIM], metric="euclidean"
    ).astype(np.float32)
    scalars = cdist(
        shapes[:, SILHOUETTE_DIM:GEOMETRY_DIM],
        shapes[:, SILHOUETTE_DIM:GEOMETRY_DIM],
        metric="euclidean",
    ).astype(np.float32)
    hog_vectors = shapes[:, GEOMETRY_DIM:]

    raw_start = perf_counter()
    raw_hog = cdist(hog_vectors, hog_vectors, metric="euclidean").astype(np.float32)
    raw_seconds = perf_counter() - raw_start
    raw_shape = normalized_group(
        settings.shape_silhouette_weight * silhouette
        + settings.shape_scalar_weight * scalars
        + settings.shape_hog_weight * raw_hog
    )
    baseline_distance = fuse(color, texture, raw_shape, settings)
    baseline = {
        "hog_dimensions": int(hog_vectors.shape[1]),
        "total_dimensions": int(colors.shape[1] + textures.shape[1] + shapes.shape[1]),
        "pairwise_hog_seconds": raw_seconds,
        "calibration": asdict(evaluate(baseline_distance, labels, calibration)),
        "test": asdict(evaluate(baseline_distance, labels, test)),
    }

    variants: dict[str, dict[str, dict]] = {}
    for scale_hog in (False, True):
        mode = "standardized" if scale_hog else "centered"
        fit_input = hog_vectors
        scaler: StandardScaler | None = None
        if scale_hog:
            scaler = StandardScaler().fit(hog_vectors[calibration])
            fit_input = scaler.transform(hog_vectors).astype(np.float32)

        fit_start = perf_counter()
        pca = PCA(
            n_components=max(PCA_DIMENSIONS),
            svd_solver="randomized",
            random_state=settings.random_seed,
        ).fit(fit_input[calibration])
        projected = pca.transform(fit_input).astype(np.float32)
        fit_seconds = perf_counter() - fit_start
        variants[mode] = {}

        for dimensions in PCA_DIMENSIONS:
            reduced = projected[:, :dimensions]
            pair_start = perf_counter()
            reduced_hog = cdist(reduced, reduced, metric="euclidean").astype(np.float32)
            pair_seconds = perf_counter() - pair_start
            reduced_shape = normalized_group(
                settings.shape_silhouette_weight * silhouette
                + settings.shape_scalar_weight * scalars
                + settings.shape_hog_weight * reduced_hog
            )
            distance = fuse(color, texture, reduced_shape, settings)
            variants[mode][str(dimensions)] = {
                "hog_dimensions": dimensions,
                "total_dimensions": int(
                    colors.shape[1] + textures.shape[1] + GEOMETRY_DIM + dimensions
                ),
                "explained_variance": float(pca.explained_variance_ratio_[:dimensions].sum()),
                "pca_fit_transform_seconds": fit_seconds,
                "pairwise_hog_seconds": pair_seconds,
                "calibration": asdict(evaluate(distance, labels, calibration)),
                "test": asdict(evaluate(distance, labels, test)),
            }

    print(
        f"\n{'Configuration':<22} {'Dims':>6} {'Var':>8} {'P@1':>8} "
        f"{'P@5':>8} {'P@10':>8} {'R@10':>8} {'mAP@10':>8}"
    )
    metrics = baseline["test"]
    print(
        f"{'Raw HOG':<22} {baseline['total_dimensions']:6d} {'1.0000':>8} "
        f"{metrics['p1']:8.4f} {metrics['p5']:8.4f} {metrics['p10']:8.4f} "
        f"{metrics['recall10']:8.4f} {metrics['map10']:8.4f}"
    )
    for mode, results in variants.items():
        for dimensions, payload in results.items():
            metrics = payload["test"]
            print(
                f"{f'PCA {dimensions} ({mode})':<22} {payload['total_dimensions']:6d} "
                f"{payload['explained_variance']:8.4f} {metrics['p1']:8.4f} "
                f"{metrics['p5']:8.4f} {metrics['p10']:8.4f} "
                f"{metrics['recall10']:8.4f} {metrics['map10']:8.4f}"
            )

    report = {
        "feature_version": settings.feature_version,
        "split": {
            "seed": settings.random_seed,
            "records": len(records),
            "calibration": len(calibration),
            "test": len(test),
            "pca_fit": "calibration only",
        },
        "baseline": baseline,
        "variants": variants,
    }
    report_path = settings.scaler_dir / "hog_pca_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
