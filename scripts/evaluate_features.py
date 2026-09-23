"""Reproducible group-level and deep component ablation for handcrafted CBIR."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist
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
    Metrics,
    evaluate,
    metric_key,
    pairwise_chi_square,
    robust_normalize,
    split_queries,
)


GROUP_CONFIGURATIONS = {
    "Color": ("color",),
    "Texture": ("texture",),
    "Shape": ("shape",),
    "Color + Texture": ("color", "texture"),
    "Color + Shape": ("color", "shape"),
    "Texture + Shape": ("texture", "shape"),
    "Color + Texture + Shape": ("color", "texture", "shape"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate complete CBIR feature ablations")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output JSON path (default: models/ablation_report.json)",
    )
    return parser.parse_args()


def weighted_sum(
    matrices: dict[str, np.ndarray], names: tuple[str, ...], weights: dict[str, float]
) -> tuple[np.ndarray, dict[str, float]]:
    selected = np.asarray([weights[name] for name in names], dtype=np.float64)
    if float(selected.sum()) <= 0.0:
        selected[:] = 1.0
    selected /= selected.sum()
    fused = sum(float(weight) * matrices[name] for name, weight in zip(names, selected))
    return fused, {name: float(weight) for name, weight in zip(names, selected)}


def candidate_weights(size: int):
    if size == 1:
        yield (1.0,)
    elif size == 2:
        for first in range(11):
            yield first / 10.0, (10 - first) / 10.0
    elif size == 3:
        for first in range(11):
            for second in range(11 - first):
                yield first / 10.0, second / 10.0, (10 - first - second) / 10.0
    else:
        raise ValueError("Ablation tuning supports one to three components")


def tune_components(
    normalized: dict[str, np.ndarray],
    names: tuple[str, ...],
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
) -> tuple[dict[str, float], Metrics, Metrics]:
    best_weights: tuple[float, ...] | None = None
    best_distance: np.ndarray | None = None
    best_metrics: Metrics | None = None
    for weights in candidate_weights(len(names)):
        distance = np.zeros_like(normalized[names[0]], dtype=np.float32)
        for name, weight in zip(names, weights):
            if weight > 0.0:
                distance += float(weight) * normalized[name]
        np.fill_diagonal(distance, np.inf)
        metrics = evaluate(distance, labels, calibration)
        if best_metrics is None or metric_key(metrics) > metric_key(best_metrics):
            best_weights = tuple(float(value) for value in weights)
            best_distance = distance
            best_metrics = metrics
    assert best_weights is not None and best_distance is not None and best_metrics is not None
    return (
        dict(zip(names, best_weights)),
        best_metrics,
        evaluate(best_distance, labels, test),
    )


def print_table(title: str, rows: dict[str, dict]) -> None:
    print(f"\n{title}")
    print(f"{'Configuration':<30} {'P@1':>8} {'P@5':>8} {'P@10':>8} {'R@10':>8} {'mAP@10':>8}")
    for name, payload in rows.items():
        metrics = payload["test"]
        print(
            f"{name:<30} {metrics['p1']:8.4f} {metrics['p5']:8.4f} "
            f"{metrics['p10']:8.4f} {metrics['recall10']:8.4f} {metrics['map10']:8.4f}"
        )


def main() -> int:
    args = parse_args()
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
    if len(records) < 2:
        raise RuntimeError("Ablation needs at least two labeled current-version records")

    labels = np.asarray([record.species for record in records], dtype=object)
    colors = np.asarray([record.feature.color_vector for record in records], dtype=np.float32)
    textures = np.asarray([record.feature.texture_vector for record in records], dtype=np.float32)
    shapes = np.asarray([record.feature.shape_vector for record in records], dtype=np.float32)
    calibration, test = split_queries(labels, settings.random_seed)
    print(
        f"Feature version={settings.feature_version}, records={len(records)}, "
        f"calibration={len(calibration)}, test={len(test)}"
    )
    print("Computing eight atomic distance matrices...")

    atomic_raw = {
        "global_hsv": pairwise_chi_square(colors[:, :JOINT_DIM]),
        "spatial_hsv": sum(
            pairwise_chi_square(
                colors[
                    :,
                    JOINT_DIM
                    + region * SPATIAL_REGION_DIM : JOINT_DIM
                    + (region + 1) * SPATIAL_REGION_DIM,
                ]
            )
            for region in range(SPATIAL_REGIONS)
        )
        / SPATIAL_REGIONS,
        "lbp": pairwise_chi_square(textures[:, :LBP_DIM]),
        "glcm": cdist(
            textures[:, LBP_DIM:BASE_TEXTURE_DIM],
            textures[:, LBP_DIM:BASE_TEXTURE_DIM],
            metric="euclidean",
        ).astype(np.float32),
        "gabor": cdist(
            textures[:, BASE_TEXTURE_DIM:],
            textures[:, BASE_TEXTURE_DIM:],
            metric="euclidean",
        ).astype(np.float32),
        "silhouette": cdist(
            shapes[:, :SILHOUETTE_DIM], shapes[:, :SILHOUETTE_DIM], metric="euclidean"
        ).astype(np.float32),
        "scalar": cdist(
            shapes[:, SILHOUETTE_DIM:GEOMETRY_DIM],
            shapes[:, SILHOUETTE_DIM:GEOMETRY_DIM],
            metric="euclidean",
        ).astype(np.float32),
        "hog": cdist(
            shapes[:, GEOMETRY_DIM:], shapes[:, GEOMETRY_DIM:], metric="euclidean"
        ).astype(np.float32),
    }

    color_raw = (
        settings.color_joint_weight * atomic_raw["global_hsv"]
        + settings.color_spatial_weight * atomic_raw["spatial_hsv"]
    )
    texture_raw = (
        settings.texture_lbp_weight * atomic_raw["lbp"]
        + settings.texture_glcm_weight * atomic_raw["glcm"]
        + settings.texture_gabor_weight * atomic_raw["gabor"]
    )
    shape_raw = (
        settings.shape_silhouette_weight * atomic_raw["silhouette"]
        + settings.shape_scalar_weight * atomic_raw["scalar"]
        + settings.shape_hog_weight * atomic_raw["hog"]
    )
    group_matrices = {
        "color": robust_normalize(color_raw)[0],
        "texture": robust_normalize(texture_raw)[0],
        "shape": robust_normalize(shape_raw)[0],
    }
    group_weights = {
        "color": settings.weight_color,
        "texture": settings.weight_texture,
        "shape": settings.weight_shape,
    }
    group_components = {
        "color": (
            ("global_hsv", settings.color_joint_weight),
            ("spatial_hsv", settings.color_spatial_weight),
        ),
        "texture": (
            ("lbp", settings.texture_lbp_weight),
            ("glcm", settings.texture_glcm_weight),
            ("gabor", settings.texture_gabor_weight),
        ),
        "shape": (
            ("silhouette", settings.shape_silhouette_weight),
            ("scalar", settings.shape_scalar_weight),
            ("hog", settings.shape_hog_weight),
        ),
    }

    group_results: dict[str, dict] = {}
    for display_name, names in GROUP_CONFIGURATIONS.items():
        distance, weights = weighted_sum(group_matrices, names, group_weights)
        group_results[display_name] = {
            "weights": weights,
            "test": asdict(evaluate(distance, labels, test)),
        }

    full_distance, full_weights = weighted_sum(
        group_matrices, ("color", "texture", "shape"), group_weights
    )
    leave_one_out: dict[str, dict] = {
        "Full model": {
            "removed": None,
            "weights": full_weights,
            "test": asdict(evaluate(full_distance, labels, test)),
        }
    }
    for affected_group, components in group_components.items():
        for removed, _ in components:
            remaining = [(name, weight) for name, weight in components if name != removed]
            remaining_weights = np.asarray([weight for _, weight in remaining], dtype=np.float64)
            if float(remaining_weights.sum()) <= 0.0:
                remaining_weights[:] = 1.0
            remaining_weights /= remaining_weights.sum()
            replacement_raw = sum(
                float(weight) * atomic_raw[name]
                for (name, _), weight in zip(remaining, remaining_weights)
            )
            modified_groups = dict(group_matrices)
            modified_groups[affected_group] = robust_normalize(replacement_raw)[0]
            distance, weights = weighted_sum(
                modified_groups, ("color", "texture", "shape"), group_weights
            )
            leave_one_out[f"Without {removed}"] = {
                "removed": removed,
                "affected_group": affected_group,
                "remaining_component_weights": {
                    name: float(weight) for (name, _), weight in zip(remaining, remaining_weights)
                },
                "weights": weights,
                "test": asdict(evaluate(distance, labels, test)),
            }

    atomic_normalized = {
        name: robust_normalize(matrix)[0] for name, matrix in atomic_raw.items()
    }
    deep_definitions = {
        "color": {
            "Global HSV": ("global_hsv",),
            "Spatial HSV": ("spatial_hsv",),
            "Global + Spatial": ("global_hsv", "spatial_hsv"),
        },
        "texture": {
            "LBP": ("lbp",),
            "GLCM": ("glcm",),
            "Gabor": ("gabor",),
            "LBP + GLCM": ("lbp", "glcm"),
            "LBP + Gabor": ("lbp", "gabor"),
            "GLCM + Gabor": ("glcm", "gabor"),
            "LBP + GLCM + Gabor": ("lbp", "glcm", "gabor"),
        },
        "shape": {
            "Silhouette": ("silhouette",),
            "Scalar": ("scalar",),
            "HOG": ("hog",),
            "Silhouette + Scalar": ("silhouette", "scalar"),
            "Silhouette + HOG": ("silhouette", "hog"),
            "Scalar + HOG": ("scalar", "hog"),
            "Silhouette + Scalar + HOG": ("silhouette", "scalar", "hog"),
        },
    }
    deep_results: dict[str, dict[str, dict]] = {}
    for group_name, definitions in deep_definitions.items():
        deep_results[group_name] = {}
        for display_name, names in definitions.items():
            weights, calibration_metrics, test_metrics = tune_components(
                atomic_normalized, names, labels, calibration, test
            )
            deep_results[group_name][display_name] = {
                "weights": weights,
                "calibration": asdict(calibration_metrics),
                "test": asdict(test_metrics),
            }

    print_table("Group ablation (production weights renormalized)", group_results)
    print_table("Full-pipeline leave-one-component-out", leave_one_out)
    for group_name, rows in deep_results.items():
        print_table(f"Deep {group_name.title()} ablation (weights tuned on calibration)", rows)

    report = {
        "feature_version": settings.feature_version,
        "split": {
            "seed": settings.random_seed,
            "records": len(records),
            "calibration": len(calibration),
            "test": len(test),
        },
        "method": {
            "group_ablation": "production group weights renormalized over selected groups",
            "deep_ablation": "atomic distances p95-normalized; subset weights tuned on calibration",
        },
        "group_ablation": group_results,
        "leave_one_component_out": leave_one_out,
        "deep_ablation": deep_results,
    }
    report_path = args.report or settings.scaler_dir / "ablation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
