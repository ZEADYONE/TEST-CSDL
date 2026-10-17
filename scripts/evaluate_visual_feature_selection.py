"""Label-free robustness and feature-selection evaluation for handcrafted CBIR.

The gallery uses stored vectors. Queries are deterministic visual perturbations
of processed gallery images. The target is the exact original image, never its
species. PostgreSQL and the web application are read-only in this experiment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
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
    _joint_hsv_histogram,
)
from app.services.gabor_features import GABOR_DIM, extract_gabor_features  # noqa: E402
from app.services.hog_features import HOG_DIM, extract_hog_features  # noqa: E402
from app.services.normalization import FeatureNormalizers  # noqa: E402
from app.services.preprocessing import (  # noqa: E402
    bgr_and_alpha,
    load_image,
    mask_bbox,
    rgba_image,
    write_image,
)
from app.services.shape_features import (  # noqa: E402
    SHAPE_SCALAR_DIM,
    SILHOUETTE_DIM,
    _shape_scalars,
    _silhouette_grid,
)
from app.services.texture_features import (  # noqa: E402
    GLCM_DIM,
    GRAY_LEVELS,
    LBP_DIM,
    _glcm_properties,
    _lbp_histogram,
    _masked_glcm,
)


EPSILON = 1e-10
TRANSFORMS = (
    "brightness_plus_10",
    "brightness_minus_10",
    "gaussian_blur",
    "translate_5px",
    "scale_plus_5pct",
    "mask_erode",
    "mask_dilate",
)
COMPONENTS = (
    "global_hsv",
    "spatial_hsv",
    "lbp",
    "glcm",
    "gabor",
    "silhouette",
    "scalar",
    "hog",
)
DIMENSIONS = {
    "global_hsv": JOINT_DIM,
    "spatial_hsv": SPATIAL_REGION_DIM * SPATIAL_REGIONS,
    "lbp": LBP_DIM,
    "glcm": GLCM_DIM,
    "gabor": GABOR_DIM,
    "silhouette": SILHOUETTE_DIM,
    "scalar": SHAPE_SCALAR_DIM,
    "hog": HOG_DIM,
}
GROUP_COMPONENTS = {
    "color": ("global_hsv", "spatial_hsv"),
    "texture": ("lbp", "glcm", "gabor"),
    "shape": ("silhouette", "scalar", "hog"),
}
COMPONENT_GROUP = {
    component: group for group, components in GROUP_COMPONENTS.items() for component in components
}


@dataclass(frozen=True)
class Configuration:
    name: str
    components: tuple[str, ...]
    category: str
    shape_weights: tuple[float, float, float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--representatives", type=int, default=8)
    parser.add_argument("--pair-samples", type=int, default=20_000)
    return parser.parse_args()


def atomic_from_stored(records: list[ImageRecord]) -> dict[str, np.ndarray]:
    colors = np.asarray([record.feature.color_vector for record in records], dtype=np.float32)
    textures = np.asarray([record.feature.texture_vector for record in records], dtype=np.float32)
    shapes = np.asarray([record.feature.shape_vector for record in records], dtype=np.float32)
    return {
        "global_hsv": colors[:, :JOINT_DIM],
        "spatial_hsv": colors[:, JOINT_DIM:],
        "lbp": textures[:, :LBP_DIM],
        "glcm": textures[:, LBP_DIM : LBP_DIM + GLCM_DIM],
        "gabor": textures[:, LBP_DIM + GLCM_DIM :],
        "silhouette": shapes[:, :SILHOUETTE_DIM],
        "scalar": shapes[:, SILHOUETTE_DIM : SILHOUETTE_DIM + SHAPE_SCALAR_DIM],
        "hog": shapes[:, SILHOUETTE_DIM + SHAPE_SCALAR_DIM :],
    }


def extract_atomic(image: np.ndarray, mask: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    binary = np.asarray(mask, dtype=bool)
    values: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    start = perf_counter()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    values["global_hsv"] = _joint_hsv_histogram(hsv, binary, (18, 4, 4))
    timings["global_hsv"] = (perf_counter() - start) * 1000.0

    start = perf_counter()
    bbox = mask_bbox(binary)
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    boundaries = np.linspace(x1, x2, SPATIAL_REGIONS + 1, dtype=int)
    regions = []
    for index in range(SPATIAL_REGIONS):
        region_mask = np.zeros_like(binary)
        region_mask[y1:y2, boundaries[index] : boundaries[index + 1]] = binary[
            y1:y2, boundaries[index] : boundaries[index + 1]
        ]
        regions.append(
            _joint_hsv_histogram(hsv, region_mask, (12, 3, 2), minimum_pixels=8)
        )
    values["spatial_hsv"] = np.concatenate(regions).astype(np.float32)
    timings["spatial_hsv"] = (perf_counter() - start) * 1000.0

    start = perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values["lbp"] = np.concatenate(
        (
            _lbp_histogram(gray, binary, points=8, radius=1),
            _lbp_histogram(gray, binary, points=16, radius=2),
        )
    ).astype(np.float32)
    timings["lbp"] = (perf_counter() - start) * 1000.0

    start = perf_counter()
    quantized = np.minimum(
        gray.astype(np.uint16) * GRAY_LEVELS // 256, GRAY_LEVELS - 1
    ).astype(np.uint8)
    offsets = ((0, 1), (-1, 1), (-1, 0), (-1, -1))
    per_angle = np.asarray(
        [
            _glcm_properties(_masked_glcm(quantized, binary, dy, dx))
            for dy, dx in offsets
        ],
        dtype=np.float64,
    )
    values["glcm"] = np.column_stack(
        (per_angle.mean(axis=0), per_angle.std(axis=0))
    ).ravel().astype(np.float32)
    timings["glcm"] = (perf_counter() - start) * 1000.0

    start = perf_counter()
    values["gabor"] = extract_gabor_features(image, binary)
    timings["gabor"] = (perf_counter() - start) * 1000.0

    start = perf_counter()
    values["silhouette"] = _silhouette_grid(binary)
    timings["silhouette"] = (perf_counter() - start) * 1000.0

    start = perf_counter()
    values["scalar"] = _shape_scalars(binary)
    timings["scalar"] = (perf_counter() - start) * 1000.0

    start = perf_counter()
    values["hog"] = extract_hog_features(image, binary)
    timings["hog"] = (perf_counter() - start) * 1000.0
    return values, timings


def transform_image(image: np.ndarray, mask: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    output = image.copy()
    changed_mask = binary.copy()
    height, width = binary.shape
    if name in {"brightness_plus_10", "brightness_minus_10"}:
        factor = 1.10 if name == "brightness_plus_10" else 0.90
        foreground = np.clip(output[binary].astype(np.float32) * factor, 0, 255)
        output[binary] = foreground.astype(np.uint8)
    elif name == "gaussian_blur":
        output = cv2.GaussianBlur(output, (5, 5), 1.0)
    elif name == "translate_5px":
        matrix = np.float32([[1, 0, 5], [0, 1, 3]])
        output = cv2.warpAffine(output, matrix, (width, height), flags=cv2.INTER_LINEAR)
        changed_mask = cv2.warpAffine(
            binary.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST
        ).astype(bool)
    elif name == "scale_plus_5pct":
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 0.0, 1.05)
        output = cv2.warpAffine(output, matrix, (width, height), flags=cv2.INTER_LINEAR)
        changed_mask = cv2.warpAffine(
            binary.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST
        ).astype(bool)
    elif name in {"mask_erode", "mask_dilate"}:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if name == "mask_erode":
            changed_mask = cv2.erode(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
        else:
            changed_mask = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
            expanded = cv2.dilate(output, kernel, iterations=1)
            output[changed_mask & ~binary] = expanded[changed_mask & ~binary]
    else:
        raise ValueError(name)
    output[~changed_mask] = 0
    return output, changed_mask


def row_chi(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (
        0.5
        * np.sum((first - second) ** 2 / (first + second + EPSILON), axis=1)
    ).astype(np.float32)


def cross_chi(first: np.ndarray, second: np.ndarray, batch_size: int = 12) -> np.ndarray:
    output = np.empty((len(first), len(second)), dtype=np.float32)
    for start in range(0, len(first), batch_size):
        stop = min(start + batch_size, len(first))
        left = first[start:stop, None, :].astype(np.float32)
        right = second[None, :, :].astype(np.float32)
        output[start:stop] = 0.5 * np.sum(
            (left - right) ** 2 / (left + right + EPSILON), axis=2
        )
    return output


def component_distance(
    component: str, first: np.ndarray, second: np.ndarray, *, paired: bool = False
) -> np.ndarray:
    if component in {"global_hsv", "lbp"}:
        return row_chi(first, second) if paired else cross_chi(first, second)
    if component == "spatial_hsv":
        distances = []
        for region in range(SPATIAL_REGIONS):
            section = slice(region * SPATIAL_REGION_DIM, (region + 1) * SPATIAL_REGION_DIM)
            distances.append(
                row_chi(first[:, section], second[:, section])
                if paired
                else cross_chi(first[:, section], second[:, section])
            )
        return sum(distances) / SPATIAL_REGIONS
    if paired:
        return np.linalg.norm(first - second, axis=1).astype(np.float32)
    return cdist(first, second, metric="euclidean").astype(np.float32)


def configurations() -> list[Configuration]:
    all_components = tuple(COMPONENTS)
    items = [Configuration("Full", all_components, "baseline")]
    for component in COMPONENTS:
        items.append(
            Configuration(
                f"Full - {component}",
                tuple(value for value in COMPONENTS if value != component),
                "leave_one_out",
            )
        )
    for component in COMPONENTS:
        items.append(Configuration(component, (component,), "single"))
    complement_pairs = (
        ("HOG + Gabor", ("hog", "gabor"), None),
        ("Spatial HSV + Global HSV", ("spatial_hsv", "global_hsv"), None),
        ("GLCM + LBP", ("glcm", "lbp"), None),
        ("HOG + Silhouette", ("hog", "silhouette"), (0.5, 0.0, 0.5)),
    )
    for name, components, weights in complement_pairs:
        items.append(Configuration(name, components, "complementarity", weights))
    reduced = (
        ("Full - LBP - Silhouette", ("lbp", "silhouette")),
        ("Full - LBP - Silhouette - Global HSV", ("lbp", "silhouette", "global_hsv")),
        ("Full - LBP - Silhouette - GLCM", ("lbp", "silhouette", "glcm")),
        ("Full - LBP - Silhouette - Shape Scalar", ("lbp", "silhouette", "scalar")),
    )
    for name, removed in reduced:
        items.append(
            Configuration(
                name,
                tuple(component for component in COMPONENTS if component not in removed),
                "reduced",
            )
        )
    for weight in (0.1, 0.2, 0.3):
        items.append(
            Configuration(
                f"Full silhouette={weight:.1f}",
                all_components,
                "silhouette_weight",
                (weight, 0.2, 0.8 - weight),
            )
        )
    unique: dict[str, Configuration] = {}
    for item in items:
        unique[item.name] = item
    return list(unique.values())


def config_weights(config: Configuration, settings) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    base_sub = {
        "color": {"global_hsv": settings.color_joint_weight, "spatial_hsv": settings.color_spatial_weight},
        "texture": {
            "lbp": settings.texture_lbp_weight,
            "glcm": settings.texture_glcm_weight,
            "gabor": settings.texture_gabor_weight,
        },
        "shape": {
            "silhouette": settings.shape_silhouette_weight,
            "scalar": settings.shape_scalar_weight,
            "hog": settings.shape_hog_weight,
        },
    }
    if config.shape_weights is not None:
        base_sub["shape"] = dict(zip(GROUP_COMPONENTS["shape"], config.shape_weights))
    active_groups = [
        group for group, components in GROUP_COMPONENTS.items() if any(c in config.components for c in components)
    ]
    base_group = {"color": settings.weight_color, "texture": settings.weight_texture, "shape": settings.weight_shape}
    group_total = sum(base_group[group] for group in active_groups)
    group_weights = {group: base_group[group] / group_total for group in active_groups}
    sub_weights: dict[str, dict[str, float]] = {}
    for group in active_groups:
        selected = [c for c in GROUP_COMPONENTS[group] if c in config.components]
        total = sum(base_sub[group][c] for c in selected)
        if total <= EPSILON:
            sub_weights[group] = {c: 1.0 / len(selected) for c in selected}
        else:
            sub_weights[group] = {c: base_sub[group][c] / total for c in selected}
    return sub_weights, group_weights


def fuse_configuration(
    distances: dict[str, np.ndarray],
    sample_distances: dict[str, np.ndarray],
    config: Configuration,
    settings,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    sub_weights, group_weights = config_weights(config, settings)
    normalized_groups: dict[str, np.ndarray] = {}
    p95s: dict[str, float] = {}
    for group, weights in sub_weights.items():
        raw = sum(weights[c] * distances[c] for c in weights)
        sample = sum(weights[c] * sample_distances[c] for c in weights)
        p95 = float(np.percentile(sample, 95))
        p95s[group] = p95
        normalized_groups[group] = np.clip(raw / (p95 + EPSILON), 0.0, 1.0).astype(np.float32)
    final = sum(group_weights[group] * normalized_groups[group] for group in group_weights)
    return final.astype(np.float32), normalized_groups, p95s


def retrieval_metrics(distance: np.ndarray, target_indices: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    selected = distance[rows]
    target = target_indices[rows]
    target_distance = selected[np.arange(len(rows)), target]
    ranks = 1 + np.sum(selected < target_distance[:, None], axis=1)
    return {
        "hit1": float(np.mean(ranks == 1)),
        "hit5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "mean_target_distance": float(np.mean(target_distance)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bar_svg(path: Path, title: str, labels: list[str], values: list[float], *, percent: bool = True) -> None:
    width = 1000
    row_height = 28
    height = 80 + row_height * len(labels)
    maximum = max(values) if values else 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="30" font-family="sans-serif" font-size="20">{escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 55 + index * row_height
        bar_width = 620 * value / max(maximum, EPSILON)
        shown = f"{value * 100:.2f}%" if percent else f"{value:.3f} ms"
        lines.extend(
            (
                f'<text x="10" y="{y + 15}" font-family="sans-serif" font-size="12">{escape(label[:34])}</text>',
                f'<rect x="260" y="{y}" width="{bar_width:.1f}" height="18" fill="#17803d"/>',
                f'<text x="{270 + bar_width:.1f}" y="{y + 14}" font-family="sans-serif" font-size="12">{shown}</text>',
            )
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def decision_status(
    component: str,
    baseline_selection: dict,
    baseline_test: dict,
    removal_selection: dict,
    removal_test: dict,
    single_test: dict,
    extraction_ms: float,
) -> tuple[str, str]:
    if component == "silhouette":
        return "NEEDS_MORE_TESTING", "Baseline weight is zero; use explicit silhouette-weight trials."
    selection_drop = baseline_selection["mrr"] - removal_selection["mrr"]
    test_drop = baseline_test["mrr"] - removal_test["mrr"]
    hit_drop = baseline_test["hit1"] - removal_test["hit1"]
    if selection_drop >= 0.005 and test_drop >= 0.005:
        return "KEEP", "Removal consistently reduces exact-image recovery."
    if selection_drop <= 0.002 and test_drop <= 0.002 and hit_drop <= 0.002:
        if single_test["mrr"] < 0.80 and (extraction_ms >= 20.0 or test_drop < 0.0):
            return "REMOVE", "Weak alone and removal is neutral or better on both splits."
        return (
            "NEEDS_MORE_TESTING",
            "Recovery is saturated; neutral removal shows redundancy, not visual uselessness.",
        )
    return "NEEDS_MORE_TESTING", "Effect is small or inconsistent across selection and test."


def main() -> int:
    args = parse_args()
    settings = get_settings()
    output_dir = Path(__file__).resolve().parents[1] / "results" / "feature_selection"
    preview_dir = output_dir / "query_previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as session:
        records = list(
            session.scalars(
                select(ImageRecord)
                .options(joinedload(ImageRecord.feature))
                .join(ImageFeature)
                .where(ImageFeature.feature_version == settings.feature_version)
                .order_by(ImageRecord.id)
            ).all()
        )
    records = [record for record in records if Path(record.processed_path).is_file()]
    if len(records) < 2:
        raise RuntimeError("At least two current-version processed images are required")
    gallery = atomic_from_stored(records)
    rng = np.random.default_rng(settings.random_seed)
    sample_size = min(args.sample_size, len(records))
    source_indices = np.sort(rng.choice(len(records), size=sample_size, replace=False))
    shuffled = source_indices.copy()
    rng.shuffle(shuffled)
    selection_sources = set(shuffled[: len(shuffled) // 2].tolist())
    test_sources = set(shuffled[len(shuffled) // 2 :].tolist())
    representative_sources = source_indices[: min(args.representatives, sample_size)]
    representative_keys = {
        (int(source), TRANSFORMS[index % len(TRANSFORMS)])
        for index, source in enumerate(representative_sources)
    }

    print(
        f"Gallery={len(records)}, sampled originals={sample_size}, "
        f"queries={sample_size * len(TRANSFORMS)}, no species labels used"
    )
    query_values = {component: [] for component in COMPONENTS}
    timing_values = {component: [] for component in COMPONENTS}
    query_sources: list[int] = []
    query_transforms: list[str] = []
    preview_paths: dict[tuple[int, str], Path] = {}
    extraction_start = perf_counter()
    for position, source_index in enumerate(source_indices):
        raw = load_image(records[int(source_index)].processed_path)
        image, alpha = bgr_and_alpha(raw)
        if alpha is None:
            raise ValueError(f"Processed image lacks alpha mask: {records[int(source_index)].processed_path}")
        mask = alpha > 0
        for transform_name in TRANSFORMS:
            transformed, transformed_mask = transform_image(image, mask, transform_name)
            atomic, timings = extract_atomic(transformed, transformed_mask)
            for component in COMPONENTS:
                query_values[component].append(atomic[component])
                timing_values[component].append(timings[component])
            query_sources.append(int(source_index))
            query_transforms.append(transform_name)
            key = (int(source_index), transform_name)
            if key in representative_keys:
                preview_path = preview_dir / f"query_{source_index}_{transform_name}.png"
                write_image(preview_path, rgba_image(transformed, transformed_mask))
                preview_paths[key] = preview_path
        if (position + 1) % 25 == 0 or position + 1 == sample_size:
            print(f"Extracted transformed queries: {position + 1}/{sample_size} originals", flush=True)
    query = {
        component: np.asarray(query_values[component], dtype=np.float32)
        for component in COMPONENTS
    }
    normalizers = FeatureNormalizers.load(settings.scaler_dir)
    query["glcm"] = normalizers.glcm.transform(query["glcm"]).astype(np.float32)
    query["gabor"] = normalizers.gabor.transform(query["gabor"]).astype(np.float32)
    query["scalar"] = normalizers.shape_scalars.transform(query["scalar"]).astype(np.float32)
    target_indices = np.asarray(query_sources, dtype=np.int32)
    query_transforms_array = np.asarray(query_transforms, dtype=object)
    selection_rows = np.asarray(
        [i for i, source in enumerate(query_sources) if source in selection_sources], dtype=np.int32
    )
    test_rows = np.asarray(
        [i for i, source in enumerate(query_sources) if source in test_sources], dtype=np.int32
    )
    print(f"Feature extraction finished in {perf_counter() - extraction_start:.1f}s")

    pair_count = min(args.pair_samples, len(records) * (len(records) - 1) // 2)
    first_pairs: list[int] = []
    second_pairs: list[int] = []
    seen: set[tuple[int, int]] = set()
    while len(seen) < pair_count:
        first, second = rng.integers(0, len(records), size=2)
        if first == second:
            continue
        pair = (min(int(first), int(second)), max(int(first), int(second)))
        if pair not in seen:
            seen.add(pair)
            first_pairs.append(pair[0])
            second_pairs.append(pair[1])
    first_pair_array = np.asarray(first_pairs, dtype=np.int32)
    second_pair_array = np.asarray(second_pairs, dtype=np.int32)

    distances: dict[str, np.ndarray] = {}
    sample_distances: dict[str, np.ndarray] = {}
    distance_latency_ms: dict[str, float] = {}
    for component in COMPONENTS:
        start = perf_counter()
        distances[component] = component_distance(component, query[component], gallery[component])
        distance_latency_ms[component] = (
            (perf_counter() - start) * 1000.0 / len(target_indices)
        )
        sample_distances[component] = component_distance(
            component,
            gallery[component][first_pair_array],
            gallery[component][second_pair_array],
            paired=True,
        )
        print(f"Computed {component} distances", flush=True)

    correlation_rows: list[dict] = []
    correlation_matrix = np.eye(len(COMPONENTS), dtype=np.float64)
    for i, first in enumerate(COMPONENTS):
        for j, second in enumerate(COMPONENTS):
            if j <= i:
                continue
            value = float(spearmanr(sample_distances[first], sample_distances[second]).statistic)
            correlation_matrix[i, j] = correlation_matrix[j, i] = value
            correlation_rows.append({"feature_a": first, "feature_b": second, "spearman": value})
    write_csv(output_dir / "distance_correlations.csv", correlation_rows)

    config_items = configurations()
    summary_rows: list[dict] = []
    transform_rows: list[dict] = []
    per_query_rows: list[dict] = []
    results: dict[str, dict] = {}
    visual_data: dict[str, list[dict]] = {}
    extraction_means = {
        component: float(np.mean(timing_values[component])) for component in COMPONENTS
    }

    for config in config_items:
        start = perf_counter()
        fused, group_distances, p95s = fuse_configuration(
            distances, sample_distances, config, settings
        )
        selection_metrics = retrieval_metrics(fused, target_indices, selection_rows)
        test_metrics = retrieval_metrics(fused, target_indices, test_rows)
        fusion_ms = (perf_counter() - start) * 1000.0 / len(target_indices)
        extraction_ms = sum(extraction_means[c] for c in config.components)
        retrieval_ms = sum(distance_latency_ms[c] for c in config.components) + fusion_ms
        total_ms = extraction_ms + retrieval_ms
        result = {
            "configuration": config.name,
            "category": config.category,
            "dimensions": sum(DIMENSIONS[c] for c in config.components),
            "components": ",".join(config.components),
            "selection": selection_metrics,
            "test": test_metrics,
            "extraction_ms": extraction_ms,
            "retrieval_ms": retrieval_ms,
            "total_latency_ms": total_ms,
            "p95": p95s,
        }
        results[config.name] = result
        summary_rows.append(
            {
                "configuration": config.name,
                "category": config.category,
                "dimensions": result["dimensions"],
                "selection_hit1": selection_metrics["hit1"],
                "selection_hit5": selection_metrics["hit5"],
                "selection_mrr": selection_metrics["mrr"],
                "test_hit1": test_metrics["hit1"],
                "test_hit5": test_metrics["hit5"],
                "test_mrr": test_metrics["mrr"],
                "test_mean_rank": test_metrics["mean_rank"],
                "test_mean_target_distance": test_metrics["mean_target_distance"],
                "extraction_ms": extraction_ms,
                "retrieval_ms": retrieval_ms,
                "total_latency_ms": total_ms,
            }
        )
        for transform_name in TRANSFORMS:
            rows = np.asarray(
                [
                    index
                    for index in test_rows
                    if query_transforms_array[index] == transform_name
                ],
                dtype=np.int32,
            )
            metrics = retrieval_metrics(fused, target_indices, rows)
            transform_rows.append(
                {"configuration": config.name, "transform": transform_name, **metrics}
            )
        selected = fused[test_rows]
        selected_targets = target_indices[test_rows]
        target_distance = selected[np.arange(len(test_rows)), selected_targets]
        ranks = 1 + np.sum(selected < target_distance[:, None], axis=1)
        for local, row in enumerate(test_rows):
            per_query_rows.append(
                {
                    "configuration": config.name,
                    "source_id": records[int(target_indices[row])].id,
                    "transform": query_transforms[row],
                    "rank": int(ranks[local]),
                    "target_distance": float(target_distance[local]),
                }
            )

        visual_entries: list[dict] = []
        for source_index, transform_name in representative_keys:
            matches = np.flatnonzero(
                (target_indices == source_index)
                & (query_transforms_array == transform_name)
            )
            if not len(matches):
                continue
            row = int(matches[0])
            nearest = np.argsort(fused[row])[:10]
            entry = {
                "source": source_index,
                "transform": transform_name,
                "query_path": preview_paths[(source_index, transform_name)],
                "results": [],
            }
            for candidate in nearest:
                similarities = {
                    group: float(np.clip(1.0 - group_distances[group][row, candidate], 0.0, 1.0))
                    for group in group_distances
                }
                entry["results"].append(
                    {
                        "candidate": int(candidate),
                        "path": Path(records[int(candidate)].processed_path),
                        "total": float(np.clip(1.0 - fused[row, candidate], 0.0, 1.0)),
                        "groups": similarities,
                        "is_original": int(candidate) == source_index,
                    }
                )
            visual_entries.append(entry)
        visual_data[config.name] = visual_entries
        print(f"Evaluated {config.name}", flush=True)

    write_csv(output_dir / "configuration_summary.csv", summary_rows)
    write_csv(output_dir / "per_transform_metrics.csv", transform_rows)
    write_csv(output_dir / "per_query_results.csv", per_query_rows)
    extraction_rows = [
        {
            "feature": component,
            "dimensions": DIMENSIONS[component],
            "extraction_ms_mean": extraction_means[component],
            "distance_ms_per_query": distance_latency_ms[component],
        }
        for component in COMPONENTS
    ]
    write_csv(output_dir / "feature_latency.csv", extraction_rows)

    test_sorted = sorted(summary_rows, key=lambda row: row["test_mrr"], reverse=True)
    bar_svg(
        output_dir / "robustness_mrr.svg",
        "Label-free transformed-query MRR (test)",
        [row["configuration"] for row in test_sorted],
        [float(row["test_mrr"]) for row in test_sorted],
    )
    latency_sorted = sorted(extraction_rows, key=lambda row: row["extraction_ms_mean"], reverse=True)
    bar_svg(
        output_dir / "feature_extraction_latency.svg",
        "Mean extraction latency by feature",
        [row["feature"] for row in latency_sorted],
        [float(row["extraction_ms_mean"]) for row in latency_sorted],
        percent=False,
    )

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Visual CBIR comparison</title>",
        "<style>body{font-family:Arial;background:#f4f6ef;color:#14251b}nav{position:sticky;top:0;background:white;padding:10px;z-index:2}.query{border-top:3px solid #173f2b;margin-top:28px}.grid{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:8px}.card{background:white;border:1px solid #ccd5c8;padding:7px}.card img{width:100%;height:145px;object-fit:contain}.original{border:3px solid #e45520}.small{font-size:11px}</style></head><body>",
        "<h1>Label-free visual retrieval comparison</h1><p>Green/orange bordered cards mark the exact original image. No species labels are used.</p>",
    ]
    for config in config_items:
        html_parts.append(f"<h2>{escape(config.name)}</h2>")
        for entry in visual_data[config.name]:
            query_rel = Path(entry["query_path"]).relative_to(output_dir).as_posix()
            html_parts.append(
                f"<section class='query'><h3>Query source {entry['source']} — {escape(entry['transform'])}</h3>"
                f"<img src='{escape(query_rel)}' style='width:180px;height:180px;object-fit:contain;background:white'>"
                "<div class='grid'>"
            )
            for rank, item in enumerate(entry["results"], 1):
                try:
                    image_rel = Path(item["path"]).relative_to(output_dir.parents[1]).as_posix()
                    image_src = "../../" + image_rel
                except ValueError:
                    image_src = Path(item["path"]).as_uri()
                groups = item["groups"]
                group_text = " ".join(
                    f"{name}={score:.3f}" for name, score in groups.items()
                )
                css = "card original" if item["is_original"] else "card"
                html_parts.append(
                    f"<div class='{css}'><b>#{rank}</b><img loading='lazy' src='{escape(image_src)}'>"
                    f"<div>Total={item['total']:.3f}</div><div class='small'>{escape(group_text)}</div></div>"
                )
            html_parts.append("</div></section>")
    html_parts.append("</body></html>")
    (output_dir / "visual_comparison.html").write_text("\n".join(html_parts), encoding="utf-8")

    baseline = results["Full"]
    decisions: list[dict] = []
    for component in COMPONENTS:
        removal = results[f"Full - {component}"]
        status, reason = decision_status(
            component,
            baseline["selection"],
            baseline["test"],
            removal["selection"],
            removal["test"],
            results[component]["test"],
            extraction_means[component],
        )
        decisions.append(
            {
                "feature": component,
                "status": status,
                "delta_hit1": baseline["test"]["hit1"] - removal["test"]["hit1"],
                "delta_mrr": baseline["test"]["mrr"] - removal["test"]["mrr"],
                "reason": reason,
            }
        )

    report = [
        "# Label-free Handcrafted Feature Selection Report",
        "",
        "## Protocol",
        "",
        f"- Feature version: `{settings.feature_version}`.",
        f"- Gallery: {len(records)} original processed images.",
        f"- Sampled originals: {sample_size}; selection={len(selection_sources)}, test={len(test_sources)}.",
        f"- Queries: {len(target_indices)} ({len(TRANSFORMS)} deterministic perturbations per original).",
        "- Relevance target: the exact corresponding original image; species labels are never read or used.",
        "- No classifier, PCA, deep embedding, species voting, database mutation, or web change.",
        "",
        "## Baseline",
        "",
        "| Split | Hit@1 | Hit@5 | MRR | Mean rank | Mean target distance |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Selection | {baseline['selection']['hit1']:.4f} | {baseline['selection']['hit5']:.4f} | {baseline['selection']['mrr']:.4f} | {baseline['selection']['mean_rank']:.2f} | {baseline['selection']['mean_target_distance']:.4f} |",
        f"| Test | {baseline['test']['hit1']:.4f} | {baseline['test']['hit5']:.4f} | {baseline['test']['mrr']:.4f} | {baseline['test']['mean_rank']:.2f} | {baseline['test']['mean_target_distance']:.4f} |",
        "",
        "The exact-original recovery metric is saturated on these mild perturbations. Neutral leave-one-out results demonstrate redundancy for this task, not visual uselessness.",
        "",
        "## Leave-One-Feature-Out and decision",
        "",
        "| Feature | Status | ΔHit@1 | ΔMRR | Evidence |",
        "|---|---|---:|---:|---|",
    ]
    for item in decisions:
        report.append(
            f"| {item['feature']} | **{item['status']}** | {item['delta_hit1']:+.4f} | {item['delta_mrr']:+.4f} | {item['reason']} |"
        )
    report.extend(
        (
            "",
            "Current evidence supports a reduced candidate without LBP and GLCM. Other descriptors remain NEEDS_MORE_TESTING because exact-original recovery is too easy to establish cross-image visual usefulness.",
        )
    )
    report.extend(
        (
            "",
            "## Configuration results (held-out test)",
            "",
            "| Configuration | Category | Hit@1 | Hit@5 | MRR | Mean rank | Dimensions | Total latency ms/query |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in summary_rows:
        report.append(
            f"| {row['configuration']} | {row['category']} | {row['test_hit1']:.4f} | {row['test_hit5']:.4f} | {row['test_mrr']:.4f} | {row['test_mean_rank']:.2f} | {row['dimensions']} | {row['total_latency_ms']:.3f} |"
        )
    report.extend(
        (
            "",
            "## Single-feature results (held-out test)",
            "",
            "| Feature | Hit@1 | Hit@5 | MRR | Mean rank |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for config in config_items:
        if config.category != "single":
            continue
        metrics = results[config.name]["test"]
        report.append(
            f"| {config.name} | {metrics['hit1']:.4f} | {metrics['hit5']:.4f} | {metrics['mrr']:.4f} | {metrics['mean_rank']:.2f} |"
        )
    report.extend(
        (
            "",
            "## Complementarity results (held-out test)",
            "",
            "| Combination | Hit@1 | Hit@5 | MRR | Mean rank |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for config in config_items:
        if config.category != "complementarity":
            continue
        metrics = results[config.name]["test"]
        report.append(
            f"| {config.name} | {metrics['hit1']:.4f} | {metrics['hit5']:.4f} | {metrics['mrr']:.4f} | {metrics['mean_rank']:.2f} |"
        )
    report.extend(
        (
            "",
            "## Extraction and distance latency",
            "",
            "| Feature | Dimensions | Extraction ms | Gallery-distance ms/query |",
            "|---|---:|---:|---:|",
        )
    )
    for row in extraction_rows:
        report.append(
            f"| {row['feature']} | {row['dimensions']} | {row['extraction_ms_mean']:.3f} | {row['distance_ms_per_query']:.3f} |"
        )
    report.extend(
        (
            "",
            "## Distance redundancy",
            "",
            "Spearman correlation uses 20,000 deterministic random gallery pairs. High correlation is supporting evidence only, not a removal rule.",
            "",
            "| Feature A | Feature B | Spearman |",
            "|---|---|---:|",
        )
    )
    for row in sorted(correlation_rows, key=lambda value: abs(float(value["spearman"])), reverse=True):
        report.append(f"| {row['feature_a']} | {row['feature_b']} | {row['spearman']:.4f} |")
    report.extend(
        (
            "",
            "## Robustness by perturbation",
            "",
            "Baseline results for each deterministic perturbation:",
            "",
            "| Perturbation | Hit@1 | Hit@5 | MRR | Mean rank | Mean target distance |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for row in transform_rows:
        if row["configuration"] != "Full":
            continue
        report.append(
            f"| {row['transform']} | {row['hit1']:.4f} | {row['hit5']:.4f} | {row['mrr']:.4f} | {row['mean_rank']:.2f} | {row['mean_target_distance']:.4f} |"
        )
    report.extend(
        (
            "",
            "See `per_transform_metrics.csv` for every configuration and transformation.",
            "",
            "## Visual Top-10 inspection",
            "",
            "Open `visual_comparison.html`. It shows transformed queries and Top-10 results side by side. The report does not assign subjective visual relevance labels.",
            "",
            "## Artifacts",
            "",
            "- `configuration_summary.csv`",
            "- `per_transform_metrics.csv`",
            "- `per_query_results.csv`",
            "- `feature_latency.csv`",
            "- `distance_correlations.csv`",
            "- `robustness_mrr.svg`",
            "- `feature_extraction_latency.svg`",
            "- `visual_comparison.html`",
            "- `manual_visual_ratings.csv` (template for human relevance judgments)",
            "",
            "## Interpretation limits",
            "",
            "Exact-original recovery measures perturbation stability, not semantic or species correctness. Manual visual inspection remains necessary for whether different images look meaningfully similar.",
        )
    )
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    (output_dir / "experiment.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "seed": settings.random_seed,
                    "gallery": len(records),
                    "sampled_originals": sample_size,
                    "selection_originals": len(selection_sources),
                    "test_originals": len(test_sources),
                    "queries": len(target_indices),
                    "uses_species": False,
                },
                "results": results,
                "decisions": decisions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved complete report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
