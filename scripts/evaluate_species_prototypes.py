"""Evaluate leave-one-out species centroids and prototype-based soft reranking."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import balanced_accuracy_score, f1_score
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
    metric_key,
    pairwise_chi_square,
    robust_normalize,
    split_queries,
)


EPSILON = 1e-10
# Keep a non-zero visual component so images inside the same species remain
# ordered by their actual appearance rather than by an arbitrary database tie.
ALPHAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def cross_chi_square(first: np.ndarray, second: np.ndarray, batch_size: int = 32) -> np.ndarray:
    output = np.empty((len(first), len(second)), dtype=np.float32)
    for start in range(0, len(first), batch_size):
        stop = min(start + batch_size, len(first))
        left = first[start:stop, None, :].astype(np.float32)
        right = second[None, :, :].astype(np.float32)
        output[start:stop] = 0.5 * np.sum(
            (left - right) ** 2 / (left + right + EPSILON), axis=2
        )
    return output


def row_chi_square(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (
        0.5
        * np.sum(
            (first.astype(np.float32) - second.astype(np.float32)) ** 2
            / (first.astype(np.float32) + second.astype(np.float32) + EPSILON),
            axis=1,
        )
    ).astype(np.float32)


def group_raw_distances(first, second, settings, *, paired: bool = False):
    chi = row_chi_square if paired else cross_chi_square
    euclidean = (
        (lambda left, right: np.linalg.norm(left - right, axis=1).astype(np.float32))
        if paired
        else (lambda left, right: cdist(left, right, metric="euclidean").astype(np.float32))
    )

    first_color, first_texture, first_shape = first
    second_color, second_texture, second_shape = second
    joint = chi(first_color[:, :JOINT_DIM], second_color[:, :JOINT_DIM])
    spatial = sum(
        chi(
            first_color[
                :,
                JOINT_DIM
                + region * SPATIAL_REGION_DIM : JOINT_DIM
                + (region + 1) * SPATIAL_REGION_DIM,
            ],
            second_color[
                :,
                JOINT_DIM
                + region * SPATIAL_REGION_DIM : JOINT_DIM
                + (region + 1) * SPATIAL_REGION_DIM,
            ],
        )
        for region in range(SPATIAL_REGIONS)
    ) / SPATIAL_REGIONS
    color = settings.color_joint_weight * joint + settings.color_spatial_weight * spatial

    lbp = chi(first_texture[:, :LBP_DIM], second_texture[:, :LBP_DIM])
    glcm = euclidean(
        first_texture[:, LBP_DIM:BASE_TEXTURE_DIM],
        second_texture[:, LBP_DIM:BASE_TEXTURE_DIM],
    )
    gabor = euclidean(first_texture[:, BASE_TEXTURE_DIM:], second_texture[:, BASE_TEXTURE_DIM:])
    texture = (
        settings.texture_lbp_weight * lbp
        + settings.texture_glcm_weight * glcm
        + settings.texture_gabor_weight * gabor
    )

    silhouette = euclidean(
        first_shape[:, :SILHOUETTE_DIM], second_shape[:, :SILHOUETTE_DIM]
    )
    scalar = euclidean(
        first_shape[:, SILHOUETTE_DIM:GEOMETRY_DIM],
        second_shape[:, SILHOUETTE_DIM:GEOMETRY_DIM],
    )
    hog = euclidean(first_shape[:, GEOMETRY_DIM:], second_shape[:, GEOMETRY_DIM:])
    shape = (
        settings.shape_silhouette_weight * silhouette
        + settings.shape_scalar_weight * scalar
        + settings.shape_hog_weight * hog
    )
    return color, texture, shape


def normalize_and_fuse(groups, p95, settings):
    color = np.clip(groups[0] / (p95[0] + EPSILON), 0.0, 1.0)
    texture = np.clip(groups[1] / (p95[1] + EPSILON), 0.0, 1.0)
    shape = np.clip(groups[2] / (p95[2] + EPSILON), 0.0, 1.0)
    return (
        settings.weight_color * color
        + settings.weight_texture * texture
        + settings.weight_shape * shape
    ).astype(np.float32)


def classification_metrics(
    prototype_distance: np.ndarray,
    label_indices: np.ndarray,
    queries: np.ndarray,
) -> dict[str, float]:
    order = np.argsort(prototype_distance[queries], axis=1)
    truth = label_indices[queries]
    predictions = order[:, 0]
    return {
        "top1_accuracy": float(np.mean(predictions == truth)),
        "top3_accuracy": float(np.mean(np.any(order[:, :3] == truth[:, None], axis=1))),
        "top5_accuracy": float(np.mean(np.any(order[:, :5] == truth[:, None], axis=1))),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
    }


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
    labels = np.asarray([str(record.species) for record in records], dtype=object)
    species = np.asarray(sorted(set(labels.tolist())), dtype=object)
    species_to_index = {name: index for index, name in enumerate(species)}
    label_indices = np.asarray([species_to_index[name] for name in labels], dtype=np.int32)
    colors = np.asarray([record.feature.color_vector for record in records], dtype=np.float32)
    textures = np.asarray([record.feature.texture_vector for record in records], dtype=np.float32)
    shapes = np.asarray([record.feature.shape_vector for record in records], dtype=np.float32)
    features = colors, textures, shapes
    calibration, test = split_queries(labels, settings.random_seed)
    print(
        f"Feature version={settings.feature_version}, records={len(records)}, "
        f"species={len(species)}, calibration={len(calibration)}, test={len(test)}"
    )

    counts = np.bincount(label_indices, minlength=len(species)).astype(np.float32)
    centroid_features = []
    loo_features = []
    for matrix in features:
        sums = np.zeros((len(species), matrix.shape[1]), dtype=np.float32)
        np.add.at(sums, label_indices, matrix)
        centroids = sums / counts[:, None]
        centroid_features.append(centroids)
        denominators = np.maximum(1.0, counts[label_indices] - 1.0)[:, None]
        loo = (sums[label_indices] - matrix) / denominators
        loo_features.append(loo.astype(np.float32))

    print("Computing visual baseline and prototype distances...")
    visual_groups = group_raw_distances(features, features, settings)
    visual_ranges = [robust_normalize(group)[1]["p95"] for group in visual_groups]
    visual_distance = normalize_and_fuse(visual_groups, visual_ranges, settings)
    np.fill_diagonal(visual_distance, np.inf)

    prototype_groups = group_raw_distances(features, tuple(centroid_features), settings)
    prototype_distance = normalize_and_fuse(
        prototype_groups, visual_ranges, settings
    )
    own_loo_groups = group_raw_distances(
        features, tuple(loo_features), settings, paired=True
    )
    own_loo_distance = normalize_and_fuse(own_loo_groups, visual_ranges, settings)
    valid_loo = counts[label_indices] > 1
    rows = np.arange(len(records))[valid_loo]
    prototype_distance[rows, label_indices[valid_loo]] = own_loo_distance[valid_loo]

    prototype_metrics = {
        "calibration": classification_metrics(
            prototype_distance, label_indices, calibration
        ),
        "test": classification_metrics(prototype_distance, label_indices, test),
    }

    candidate_species_distance = prototype_distance[:, label_indices]
    reranking: dict[str, dict] = {}
    best_map_alpha: float | None = None
    best_map_calibration = None
    for alpha in ALPHAS:
        final_distance = (
            alpha * visual_distance + (1.0 - alpha) * candidate_species_distance
        ).astype(np.float32)
        np.fill_diagonal(final_distance, np.inf)
        calibration_metrics = evaluate(final_distance, labels, calibration)
        test_metrics = evaluate(final_distance, labels, test)
        reranking[f"{alpha:.1f}"] = {
            "visual_weight": alpha,
            "prototype_weight": 1.0 - alpha,
            "calibration": asdict(calibration_metrics),
            "test": asdict(test_metrics),
        }
        if best_map_calibration is None or metric_key(calibration_metrics) > metric_key(
            best_map_calibration
        ):
            best_map_alpha = alpha
            best_map_calibration = calibration_metrics

    best_p1_alpha = max(
        ALPHAS,
        key=lambda alpha: (
            reranking[f"{alpha:.1f}"]["calibration"]["p1"],
            alpha,
            reranking[f"{alpha:.1f}"]["calibration"]["map10"],
        ),
    )
    # This policy keeps visual similarity at least as influential as species.
    balanced_candidates = tuple(alpha for alpha in ALPHAS if alpha >= 0.5)
    balanced_alpha = max(
        balanced_candidates,
        key=lambda alpha: metric_key(
            evaluate(
                (
                    alpha * visual_distance
                    + (1.0 - alpha) * candidate_species_distance
                ).astype(np.float32),
                labels,
                calibration,
            )
        ),
    )

    print("\nNearest-centroid species prediction")
    print(json.dumps(prototype_metrics, indent=2))
    print(
        f"\n{'Alpha':>7} {'P@1':>8} {'P@5':>8} {'P@10':>8} "
        f"{'R@10':>8} {'mAP@10':>8}"
    )
    for alpha, payload in reranking.items():
        metrics = payload["test"]
        markers = ""
        if float(alpha) == best_map_alpha:
            markers += "M"
        if float(alpha) == best_p1_alpha:
            markers += "P"
        if float(alpha) == balanced_alpha:
            markers += "B"
        suffix = f" {markers}" if markers else ""
        print(
            f"{alpha + suffix:>7} {metrics['p1']:8.4f} {metrics['p5']:8.4f} "
            f"{metrics['p10']:8.4f} {metrics['recall10']:8.4f} {metrics['map10']:8.4f}"
        )
    print("M=max calibration mAP@10; P=max calibration P@1; B=balanced policy")

    report = {
        "feature_version": settings.feature_version,
        "protocol": "all labeled gallery centroids; own-species centroid excludes current query",
        "split": {
            "seed": settings.random_seed,
            "records": len(records),
            "species": len(species),
            "calibration": len(calibration),
            "test": len(test),
        },
        "species_counts": {
            "minimum": int(counts.min()),
            "median": float(np.median(counts)),
            "maximum": int(counts.max()),
        },
        "prototype_prediction": prototype_metrics,
        "selected_alpha_for_map10": best_map_alpha,
        "selected_alpha_for_p1": best_p1_alpha,
        "recommended_balanced_alpha": balanced_alpha,
        "reranking": reranking,
    }
    report_path = settings.scaler_dir / "species_prototype_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
