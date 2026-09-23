"""Tune handcrafted CBIR weights on a stratified calibration/test split.

The script uses only vectors already stored in PostgreSQL. It does not alter
the database or configuration; it prints reproducible recommendations and
writes a JSON experiment report under ``models/``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
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
from app.services.color_features import JOINT_DIM, SPATIAL_REGION_DIM, SPATIAL_REGIONS  # noqa: E402
from app.services.shape_features import SHAPE_DIM as GEOMETRY_DIM, SILHOUETTE_DIM  # noqa: E402
from app.services.texture_features import LBP_DIM, TEXTURE_DIM as BASE_TEXTURE_DIM  # noqa: E402


EPSILON = 1e-10


@dataclass(frozen=True)
class Metrics:
    p1: float
    p5: float
    p10: float
    recall10: float
    map10: float


def pairwise_chi_square(vectors: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """Compute an NxN chi-square matrix in bounded-memory batches."""

    count = len(vectors)
    output = np.empty((count, count), dtype=np.float32)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        first = vectors[start:stop, None, :].astype(np.float32)
        second = vectors[None, :, :].astype(np.float32)
        output[start:stop] = 0.5 * np.sum(
            (first - second) ** 2 / (first + second + EPSILON), axis=2
        )
    return output


def robust_normalize(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    upper = matrix[np.triu_indices(len(matrix), 1)]
    p5, p95 = np.percentile(upper, (5, 95))
    normalized = np.clip(matrix / (float(p95) + EPSILON), 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(normalized, np.inf)
    return normalized, {"p5": float(p5), "p95": float(p95)}


def evaluate(distance: np.ndarray, labels: np.ndarray, queries: np.ndarray) -> Metrics:
    totals = defaultdict(float)
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[str(label)] += 1
    for query_index in queries:
        row = distance[query_index]
        nearest = np.argpartition(row, 10)[:10]
        nearest = nearest[np.argsort(row[nearest])]
        relevant = labels[nearest] == labels[query_index]
        totals["p1"] += float(relevant[:1].sum())
        totals["p5"] += float(relevant[:5].sum()) / 5.0
        totals["p10"] += float(relevant.sum()) / 10.0
        relevant_total = max(1, label_counts[str(labels[query_index])] - 1)
        totals["recall10"] += float(relevant.sum()) / relevant_total
        cumulative = np.cumsum(relevant)
        precisions = cumulative / np.arange(1, 11)
        totals["map10"] += float(np.sum(precisions * relevant)) / min(relevant_total, 10)
    size = float(len(queries))
    return Metrics(*(totals[name] / size for name in ("p1", "p5", "p10", "recall10", "map10")))


def split_queries(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[str(label)].append(index)
    calibration: list[int] = []
    test: list[int] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        shuffled = np.asarray(indices, dtype=int)
        rng.shuffle(shuffled)
        cut = max(1, len(shuffled) // 2)
        calibration.extend(shuffled[:cut].tolist())
        test.extend(shuffled[cut:].tolist())
    return np.asarray(sorted(calibration)), np.asarray(sorted(test))


def metric_key(metrics: Metrics) -> tuple[float, float, float]:
    return metrics.map10, metrics.p1, metrics.p5


def tune_final_weights(
    groups: tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: np.ndarray,
    calibration: np.ndarray,
) -> tuple[float, float, float]:
    best = (0.4, 0.3, 0.3)
    best_metrics: Metrics | None = None
    for color_int in range(1, 9):
        for texture_int in range(1, 10 - color_int):
            shape_int = 10 - color_int - texture_int
            if shape_int < 1:
                continue
            weights = (color_int / 10.0, texture_int / 10.0, shape_int / 10.0)
            fused = sum(weight * group for weight, group in zip(weights, groups))
            metrics = evaluate(fused, labels, calibration)
            if best_metrics is None or metric_key(metrics) > metric_key(best_metrics):
                best, best_metrics = weights, metrics
    return best


def build_groups(
    atomic: tuple[np.ndarray, ...],
    component_weights: list[float],
) -> tuple[tuple[np.ndarray, ...], dict[str, dict[str, float]]]:
    joint, spatial, lbp, glcm, gabor, silhouette, scalars, hog = atomic
    (
        color_joint,
        texture_lbp,
        texture_glcm,
        texture_gabor,
        shape_silhouette,
        shape_scalar,
        shape_hog,
    ) = component_weights
    raw_groups = (
        color_joint * joint + (1.0 - color_joint) * spatial,
        texture_lbp * lbp + texture_glcm * glcm + texture_gabor * gabor,
        shape_silhouette * silhouette + shape_scalar * scalars + shape_hog * hog,
    )
    normalized_with_ranges = [robust_normalize(group) for group in raw_groups]
    groups = tuple(item[0] for item in normalized_with_ranges)
    ranges = {
        name: item[1]
        for name, item in zip(("color", "texture", "shape"), normalized_with_ranges)
    }
    return groups, ranges


def coordinate_search(
    atomic: tuple[np.ndarray, ...],
    initial_components: tuple[float, float, float, float, float, float, float],
    labels: np.ndarray,
    calibration: np.ndarray,
) -> tuple[list[float], tuple[float, float, float], tuple[np.ndarray, ...], dict[str, dict[str, float]]]:
    """Jointly improve sub-weights and final weights without decreasing calibration mAP."""

    components = list(initial_components)
    groups, ranges = build_groups(atomic, components)
    final_weights = tune_final_weights(groups, labels, calibration)
    for _ in range(3):
        changed = False
        for component_index in (0,):
            best_weight = components[component_index]
            best_components = components.copy()
            best_groups = groups
            best_ranges = ranges
            best_distance = sum(
                weight * group for weight, group in zip(final_weights, groups)
            )
            best_metrics = evaluate(best_distance, labels, calibration)
            for integer_weight in range(11):
                candidate_weight = integer_weight / 10.0
                candidate_components = components.copy()
                candidate_components[component_index] = candidate_weight
                candidate_groups, candidate_ranges = build_groups(atomic, candidate_components)
                candidate_distance = sum(
                    weight * group for weight, group in zip(final_weights, candidate_groups)
                )
                candidate_metrics = evaluate(candidate_distance, labels, calibration)
                if metric_key(candidate_metrics) > metric_key(best_metrics):
                    best_weight = candidate_weight
                    best_components = candidate_components
                    best_groups = candidate_groups
                    best_ranges = candidate_ranges
                    best_metrics = candidate_metrics
            if best_weight != components[component_index]:
                changed = True
            components = best_components
            groups = best_groups
            ranges = best_ranges

        best_texture = tuple(components[1:4])
        best_components = components.copy()
        best_groups = groups
        best_ranges = ranges
        best_metrics = evaluate(
            sum(weight * group for weight, group in zip(final_weights, groups)),
            labels,
            calibration,
        )
        for lbp_int in range(11):
            for glcm_int in range(11 - lbp_int):
                gabor_int = 10 - lbp_int - glcm_int
                candidate_texture = (lbp_int / 10.0, glcm_int / 10.0, gabor_int / 10.0)
                candidate_components = components.copy()
                candidate_components[1:4] = candidate_texture
                candidate_groups, candidate_ranges = build_groups(atomic, candidate_components)
                candidate_distance = sum(
                    weight * group for weight, group in zip(final_weights, candidate_groups)
                )
                candidate_metrics = evaluate(candidate_distance, labels, calibration)
                if metric_key(candidate_metrics) > metric_key(best_metrics):
                    best_texture = candidate_texture
                    best_components = candidate_components
                    best_groups = candidate_groups
                    best_ranges = candidate_ranges
                    best_metrics = candidate_metrics
        if best_texture != tuple(components[1:4]):
            changed = True
        components = best_components
        groups = best_groups
        ranges = best_ranges

        best_shape = tuple(components[4:7])
        best_components = components.copy()
        best_groups = groups
        best_ranges = ranges
        best_metrics = evaluate(
            sum(weight * group for weight, group in zip(final_weights, groups)),
            labels,
            calibration,
        )
        for silhouette_int in range(11):
            for scalar_int in range(11 - silhouette_int):
                hog_int = 10 - silhouette_int - scalar_int
                candidate_shape = (
                    silhouette_int / 10.0,
                    scalar_int / 10.0,
                    hog_int / 10.0,
                )
                candidate_components = components.copy()
                candidate_components[4:7] = candidate_shape
                candidate_groups, candidate_ranges = build_groups(atomic, candidate_components)
                candidate_distance = sum(
                    weight * group for weight, group in zip(final_weights, candidate_groups)
                )
                candidate_metrics = evaluate(candidate_distance, labels, calibration)
                if metric_key(candidate_metrics) > metric_key(best_metrics):
                    best_shape = candidate_shape
                    best_components = candidate_components
                    best_groups = candidate_groups
                    best_ranges = candidate_ranges
                    best_metrics = candidate_metrics
        if best_shape != tuple(components[4:7]):
            changed = True
        components = best_components
        groups = best_groups
        ranges = best_ranges

        tuned_final = tune_final_weights(groups, labels, calibration)
        changed = changed or tuned_final != final_weights
        final_weights = tuned_final
        if not changed:
            break
    return components, final_weights, groups, ranges


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
    print(f"Records={len(records)}, calibration={len(calibration)}, test={len(test)}")
    print("Computing eight atomic distance matrices...")
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
    lbp = pairwise_chi_square(textures[:, :LBP_DIM])
    glcm = cdist(
        textures[:, LBP_DIM:BASE_TEXTURE_DIM],
        textures[:, LBP_DIM:BASE_TEXTURE_DIM],
        metric="euclidean",
    ).astype(np.float32)
    gabor = cdist(
        textures[:, BASE_TEXTURE_DIM:], textures[:, BASE_TEXTURE_DIM:], metric="euclidean"
    ).astype(np.float32)
    silhouette = cdist(
        shapes[:, :SILHOUETTE_DIM], shapes[:, :SILHOUETTE_DIM], metric="euclidean"
    ).astype(np.float32)
    scalars = cdist(
        shapes[:, SILHOUETTE_DIM:GEOMETRY_DIM],
        shapes[:, SILHOUETTE_DIM:GEOMETRY_DIM],
        metric="euclidean",
    ).astype(np.float32)
    hog = cdist(shapes[:, GEOMETRY_DIM:], shapes[:, GEOMETRY_DIM:], metric="euclidean").astype(
        np.float32
    )

    baseline_components = (
        settings.color_joint_weight,
        settings.texture_lbp_weight,
        settings.texture_glcm_weight,
        settings.texture_gabor_weight,
        settings.shape_silhouette_weight,
        settings.shape_scalar_weight,
        settings.shape_hog_weight,
    )
    atomic = (joint, spatial, lbp, glcm, gabor, silhouette, scalars, hog)
    baseline_groups, _ = build_groups(atomic, list(baseline_components))
    baseline_weights = (settings.weight_color, settings.weight_texture, settings.weight_shape)
    baseline_distance = sum(
        weight * group for weight, group in zip(baseline_weights, baseline_groups)
    )

    component_weights, final_weights, tuned_groups, distance_ranges = coordinate_search(
        atomic, baseline_components, labels, calibration
    )
    tuned_distance = sum(weight * group for weight, group in zip(final_weights, tuned_groups))

    non_hog_total = settings.shape_silhouette_weight + settings.shape_scalar_weight
    without_hog_components = list(baseline_components)
    without_hog_components[4:7] = (
        settings.shape_silhouette_weight / non_hog_total,
        settings.shape_scalar_weight / non_hog_total,
        0.0,
    )
    without_hog_groups, _ = build_groups(atomic, without_hog_components)
    without_hog_distance = sum(
        weight * group for weight, group in zip(baseline_weights, without_hog_groups)
    )

    non_gabor_total = settings.texture_lbp_weight + settings.texture_glcm_weight
    without_gabor_components = list(baseline_components)
    without_gabor_components[1:4] = (
        settings.texture_lbp_weight / non_gabor_total,
        settings.texture_glcm_weight / non_gabor_total,
        0.0,
    )
    without_gabor_groups, _ = build_groups(atomic, without_gabor_components)
    without_gabor_distance = sum(
        weight * group for weight, group in zip(baseline_weights, without_gabor_groups)
    )

    baseline_calibration = evaluate(baseline_distance, labels, calibration)
    tuned_calibration = evaluate(tuned_distance, labels, calibration)
    baseline_test = evaluate(baseline_distance, labels, test)
    tuned_test = evaluate(tuned_distance, labels, test)
    without_hog_calibration = evaluate(without_hog_distance, labels, calibration)
    without_hog_test = evaluate(without_hog_distance, labels, test)
    without_gabor_calibration = evaluate(without_gabor_distance, labels, calibration)
    without_gabor_test = evaluate(without_gabor_distance, labels, test)
    report = {
        "split": {"seed": settings.random_seed, "calibration": len(calibration), "test": len(test)},
        "baseline": {
            "component_weights": baseline_components,
            "final_weights": baseline_weights,
            "calibration": asdict(baseline_calibration),
            "test": asdict(baseline_test),
        },
        "ablation_without_hog": {
            "component_weights": without_hog_components,
            "final_weights": baseline_weights,
            "calibration": asdict(without_hog_calibration),
            "test": asdict(without_hog_test),
        },
        "ablation_without_gabor": {
            "component_weights": without_gabor_components,
            "final_weights": baseline_weights,
            "calibration": asdict(without_gabor_calibration),
            "test": asdict(without_gabor_test),
        },
        "tuned": {
            "component_weights": {
                "color_joint": component_weights[0],
                "texture_lbp": component_weights[1],
                "texture_glcm": component_weights[2],
                "texture_gabor": component_weights[3],
                "shape_silhouette": component_weights[4],
                "shape_scalar": component_weights[5],
                "shape_hog": component_weights[6],
            },
            "final_weights": {
                "color": final_weights[0],
                "texture": final_weights[1],
                "shape": final_weights[2],
            },
            "distance_ranges": distance_ranges,
            "calibration": asdict(tuned_calibration),
            "test": asdict(tuned_test),
        },
    }
    report_path = settings.scaler_dir / "weight_tuning_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
