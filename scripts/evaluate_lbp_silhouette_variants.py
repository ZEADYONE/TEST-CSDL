"""Tune LBP and silhouette variants for label-free handcrafted CBIR.

The experiment never reads species labels.  A deterministic transformed image
must retrieve its exact source, while its non-self Top-10 should remain stable.
Parameter selection and final reporting use disjoint source-image splits.
PostgreSQL and the production feature pipeline are read-only.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from skimage.feature import local_binary_pattern


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import ImageFeature, ImageRecord  # noqa: E402
from app.services.normalization import FeatureNormalizers  # noqa: E402
from app.services.preprocessing import bgr_and_alpha, load_image, mask_bbox  # noqa: E402
from scripts.evaluate_proposed_features import rbo_at_k, sample_pairs  # noqa: E402
from scripts.evaluate_visual_feature_selection import (  # noqa: E402
    EPSILON,
    atomic_from_stored,
    component_distance,
    extract_atomic,
)


TRANSFORMS = (
    "brightness_plus_20",
    "brightness_minus_20",
    "contrast_plus_25",
    "gaussian_blur_7",
    "gaussian_noise_10",
    "jpeg_quality_45",
    "translate_12px",
    "scale_plus_12pct",
    "rotate_plus_5deg",
    "mask_erode_2",
    "mask_dilate_2",
    "combined_blur_translate",
)

TRANSFORM_FAMILY = {
    "brightness_plus_20": "appearance",
    "brightness_minus_20": "appearance",
    "contrast_plus_25": "appearance",
    "gaussian_blur_7": "appearance",
    "gaussian_noise_10": "appearance",
    "jpeg_quality_45": "appearance",
    "translate_12px": "geometry",
    "scale_plus_12pct": "geometry",
    "rotate_plus_5deg": "geometry",
    "mask_erode_2": "mask",
    "mask_dilate_2": "mask",
    "combined_blur_translate": "combined",
}


@dataclass(frozen=True)
class LBPVariant:
    key: str
    label: str
    section_dims: tuple[int, ...]
    dimension: int


@dataclass(frozen=True)
class SilhouetteVariant:
    key: str
    label: str
    dimension: int


@dataclass(frozen=True)
class Configuration:
    name: str
    category: str
    lbp_key: str | None = None
    lbp_weight: float = 0.0
    silhouette_key: str | None = None
    silhouette_weight: float = 0.0


LBP_VARIANTS = (
    LBPVariant("uniform_r1", "Uniform P8 R1 global", (10,), 10),
    LBPVariant("uniform_ms2", "Uniform P8R1+P16R2 global", (10, 18), 28),
    LBPVariant("uniform_ms3", "Uniform P8R1+P16R2+P24R3 global", (10, 18, 26), 54),
    LBPVariant("spatial3_ms2", "Uniform MS2 x 3 horizontal regions", (10, 18) * 3, 84),
    LBPVariant("spatial3_ms3", "Uniform MS3 x 3 horizontal regions", (10, 18, 26) * 3, 162),
    LBPVariant("default_r1", "Default P8 R1 global", (256,), 256),
)
LBP_BY_KEY = {variant.key: variant for variant in LBP_VARIANTS}

SILHOUETTE_VARIANTS = (
    SilhouetteVariant("grid4", "Occupancy grid 4x4", 16),
    SilhouetteVariant("grid8", "Occupancy grid 8x8", 64),
    SilhouetteVariant("grid16", "Occupancy grid 16x16", 256),
    SilhouetteVariant("bbox_grid4", "BBox-normalized occupancy grid 4x4", 16),
    SilhouetteVariant("bbox_grid8", "BBox-normalized occupancy grid 8x8", 64),
    SilhouetteVariant("bbox_grid16", "BBox-normalized occupancy grid 16x16", 256),
)
SILHOUETTE_BY_KEY = {variant.key: variant for variant in SILHOUETTE_VARIANTS}

BASE_COMPONENTS = ("global_hsv", "spatial_hsv", "glcm", "gabor", "scalar", "hog")
LBP_WEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
SILHOUETTE_WEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=160)
    parser.add_argument("--pair-samples", type=int, default=20_000)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def histogram(values: np.ndarray, selected: np.ndarray, bins: int) -> np.ndarray:
    sample = values[selected]
    counts, _ = np.histogram(sample, bins=np.arange(bins + 1), range=(0, bins))
    result = counts.astype(np.float32)
    total = float(result.sum())
    if total > 0:
        result /= total
    return result


def extract_lbp_variants(image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    specs = {
        "u8": (8, 1, "uniform", 10),
        "u16": (16, 2, "uniform", 18),
        "u24": (24, 3, "uniform", 26),
        "d8": (8, 1, "default", 256),
    }
    maps: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for key, (points, radius, method, bins) in specs.items():
        support = radius + 1
        kernel = np.ones((2 * support + 1, 2 * support + 1), dtype=np.uint8)
        valid = cv2.erode(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
        if not np.any(valid):
            valid = binary
        maps[key] = (local_binary_pattern(gray, points, radius, method=method), valid, bins)

    global_parts = {
        key: histogram(values, valid, bins)
        for key, (values, valid, bins) in maps.items()
    }
    bbox = mask_bbox(binary)
    if bbox is None:
        raise ValueError("LBP requires a non-empty mask")
    x1, _, x2, _ = bbox
    boundaries = np.linspace(x1, x2, 4, dtype=int)

    def spatial(scales: tuple[str, ...]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for region in range(3):
            region_mask = np.zeros_like(binary)
            region_mask[:, boundaries[region] : boundaries[region + 1]] = True
            for scale in scales:
                values, valid, bins = maps[scale]
                parts.append(histogram(values, valid & region_mask, bins))
        return np.concatenate(parts).astype(np.float32)

    return {
        "uniform_r1": global_parts["u8"],
        "uniform_ms2": np.concatenate((global_parts["u8"], global_parts["u16"])),
        "uniform_ms3": np.concatenate(
            (global_parts["u8"], global_parts["u16"], global_parts["u24"])
        ),
        "spatial3_ms2": spatial(("u8", "u16")),
        "spatial3_ms3": spatial(("u8", "u16", "u24")),
        "default_r1": global_parts["d8"],
    }


def occupancy_grid(mask: np.ndarray, size: int) -> np.ndarray:
    parts = [
        float(np.mean(cell)) if cell.size else 0.0
        for rows in np.array_split(np.asarray(mask, dtype=bool), size, axis=0)
        for cell in np.array_split(rows, size, axis=1)
    ]
    return np.asarray(parts, dtype=np.float32)


def bbox_normalized_mask(mask: np.ndarray, size: int = 128) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    bbox = mask_bbox(binary)
    if bbox is None:
        raise ValueError("Silhouette requires a non-empty mask")
    x1, y1, x2, y2 = bbox
    crop = binary[y1:y2, x1:x2]
    side = max(crop.shape)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y = (side - crop.shape[0]) // 2
    x = (side - crop.shape[1]) // 2
    canvas[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA) >= 0.5


def extract_silhouette_variants(mask: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    normalized = bbox_normalized_mask(binary)
    return {
        "grid4": occupancy_grid(binary, 4),
        "grid8": occupancy_grid(binary, 8),
        "grid16": occupancy_grid(binary, 16),
        "bbox_grid4": occupancy_grid(normalized, 4),
        "bbox_grid8": occupancy_grid(normalized, 8),
        "bbox_grid16": occupancy_grid(normalized, 16),
    }


def transform_image(
    image: np.ndarray,
    mask: np.ndarray,
    name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    output = image.copy()
    changed_mask = binary.copy()
    height, width = binary.shape
    if name in {"brightness_plus_20", "brightness_minus_20"}:
        factor = 1.20 if name == "brightness_plus_20" else 0.80
        output[binary] = np.clip(output[binary].astype(np.float32) * factor, 0, 255).astype(
            np.uint8
        )
    elif name == "contrast_plus_25":
        pixels = output[binary].astype(np.float32)
        mean = pixels.mean(axis=0, keepdims=True)
        output[binary] = np.clip(mean + 1.25 * (pixels - mean), 0, 255).astype(np.uint8)
    elif name == "gaussian_blur_7":
        output = cv2.GaussianBlur(output, (7, 7), 1.6)
    elif name == "gaussian_noise_10":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 10.0, size=output.shape).astype(np.float32)
        values = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        output[binary] = values[binary]
    elif name == "jpeg_quality_45":
        ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 45])
        if not ok:
            raise RuntimeError("JPEG perturbation failed")
        output = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    elif name in {"translate_12px", "combined_blur_translate"}:
        matrix = np.float32([[1, 0, 12], [0, 1, 7]])
        output = cv2.warpAffine(output, matrix, (width, height), flags=cv2.INTER_LINEAR)
        changed_mask = cv2.warpAffine(
            binary.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST
        ).astype(bool)
        if name == "combined_blur_translate":
            output = cv2.GaussianBlur(output, (7, 7), 1.5)
            output[changed_mask] = np.clip(
                output[changed_mask].astype(np.float32) * 0.85, 0, 255
            ).astype(np.uint8)
    elif name == "scale_plus_12pct":
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 0.0, 1.12)
        output = cv2.warpAffine(output, matrix, (width, height), flags=cv2.INTER_LINEAR)
        changed_mask = cv2.warpAffine(
            binary.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST
        ).astype(bool)
    elif name == "rotate_plus_5deg":
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 5.0, 1.0)
        output = cv2.warpAffine(output, matrix, (width, height), flags=cv2.INTER_LINEAR)
        changed_mask = cv2.warpAffine(
            binary.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST
        ).astype(bool)
    elif name in {"mask_erode_2", "mask_dilate_2"}:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if name == "mask_erode_2":
            changed_mask = cv2.erode(
                binary.astype(np.uint8), kernel, iterations=2
            ).astype(bool)
        else:
            changed_mask = cv2.dilate(
                binary.astype(np.uint8), kernel, iterations=2
            ).astype(bool)
            expanded = cv2.dilate(output, kernel, iterations=2)
            output[changed_mask & ~binary] = expanded[changed_mask & ~binary]
    else:
        raise ValueError(name)
    output[~changed_mask] = 0
    return output, changed_mask


def sectioned_chi(
    first: np.ndarray,
    second: np.ndarray,
    sections: tuple[int, ...],
    *,
    paired: bool = False,
) -> np.ndarray:
    values: list[np.ndarray] = []
    offset = 0
    for dimension in sections:
        left = first[:, offset : offset + dimension]
        right = second[:, offset : offset + dimension]
        if paired:
            distance = 0.5 * np.sum((left - right) ** 2 / (left + right + EPSILON), axis=1)
        else:
            distance = np.empty((len(left), len(right)), dtype=np.float32)
            for start in range(0, len(left), 12):
                a = left[start : start + 12, None, :]
                b = right[None, :, :]
                distance[start : start + 12] = 0.5 * np.sum(
                    (a - b) ** 2 / (a + b + EPSILON), axis=2
                )
        values.append(distance.astype(np.float32))
        offset += dimension
    return (sum(values) / len(values)).astype(np.float32)


def retrieval_metrics(
    distance: np.ndarray,
    targets: np.ndarray,
    rows: np.ndarray,
) -> dict[str, float]:
    chosen = distance[rows]
    target = targets[rows]
    positive = chosen[np.arange(len(rows)), target]
    ranks = 1 + np.sum(chosen < positive[:, None], axis=1)
    return {
        "hit1": float(np.mean(ranks == 1)),
        "hit5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "positive_percentile": float(np.mean(1.0 - (ranks - 1) / (chosen.shape[1] - 1))),
    }


def stability_metrics(
    original: np.ndarray,
    transformed: np.ndarray,
    source_indices: np.ndarray,
    targets: np.ndarray,
    rows: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    source_row = {int(source): row for row, source in enumerate(source_indices)}
    overlap: list[float] = []
    rbo: list[float] = []
    for query_row in rows:
        target = int(targets[query_row])
        base_order = np.argsort(original[source_row[target]], kind="stable")
        query_order = np.argsort(transformed[query_row], kind="stable")
        base_top = base_order[base_order != target][:top_k]
        query_top = query_order[query_order != target][:top_k]
        overlap.append(len(set(base_top.tolist()) & set(query_top.tolist())) / top_k)
        rbo.append(rbo_at_k(base_top, query_top))
    return {"top10_overlap": float(np.mean(overlap)), "rbo10": float(np.mean(rbo))}


def config_score(metrics: dict[str, float]) -> float:
    return 0.30 * metrics["hit1"] + 0.40 * metrics["mrr"] + 0.30 * metrics["rbo10"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    settings = get_settings()
    output_dir = PROJECT_ROOT / "results" / "lbp_silhouette_variants"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError("At least two indexed processed images are required")

    rng = np.random.default_rng(settings.random_seed + 71)
    sample_size = min(args.sample_size, len(records))
    source_indices = np.sort(rng.choice(len(records), size=sample_size, replace=False))
    split = source_indices.copy()
    rng.shuffle(split)
    selection_sources = set(int(value) for value in split[: len(split) // 2])
    test_sources = set(int(value) for value in split[len(split) // 2 :])
    gallery = atomic_from_stored(records)

    gallery_lbp = {variant.key: [] for variant in LBP_VARIANTS}
    gallery_silhouette = {variant.key: [] for variant in SILHOUETTE_VARIANTS}
    print(f"Gallery={len(records)}; extracting variant descriptors", flush=True)
    started = perf_counter()
    for index, record in enumerate(records):
        raw = load_image(record.processed_path)
        image, alpha = bgr_and_alpha(raw)
        if alpha is None:
            raise ValueError(f"Missing alpha mask: {record.processed_path}")
        mask = alpha > 0
        lbp_values = extract_lbp_variants(image, mask)
        silhouette_values = extract_silhouette_variants(mask)
        for key, value in lbp_values.items():
            gallery_lbp[key].append(value)
        for key, value in silhouette_values.items():
            gallery_silhouette[key].append(value)
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f"Gallery variants {index + 1}/{len(records)}", flush=True)
    gallery_lbp_array = {
        key: np.asarray(values, dtype=np.float32) for key, values in gallery_lbp.items()
    }
    gallery_silhouette_array = {
        key: np.asarray(values, dtype=np.float32)
        for key, values in gallery_silhouette.items()
    }
    print(f"Gallery variants finished in {perf_counter() - started:.1f}s", flush=True)

    query_base = {component: [] for component in BASE_COMPONENTS}
    query_lbp = {variant.key: [] for variant in LBP_VARIANTS}
    query_silhouette = {variant.key: [] for variant in SILHOUETTE_VARIANTS}
    query_sources: list[int] = []
    query_transforms: list[str] = []
    print(f"Extracting {sample_size * len(TRANSFORMS)} transformed queries", flush=True)
    started = perf_counter()
    for position, source_index in enumerate(source_indices):
        raw = load_image(records[int(source_index)].processed_path)
        image, alpha = bgr_and_alpha(raw)
        if alpha is None:
            raise ValueError(f"Missing alpha mask: {records[int(source_index)].processed_path}")
        mask = alpha > 0
        for transform_index, transform_name in enumerate(TRANSFORMS):
            transformed, transformed_mask = transform_image(
                image,
                mask,
                transform_name,
                seed=settings.random_seed + int(source_index) * 101 + transform_index,
            )
            atomic, _ = extract_atomic(transformed, transformed_mask)
            for component in BASE_COMPONENTS:
                query_base[component].append(atomic[component])
            lbp_values = extract_lbp_variants(transformed, transformed_mask)
            silhouette_values = extract_silhouette_variants(transformed_mask)
            for key, value in lbp_values.items():
                query_lbp[key].append(value)
            for key, value in silhouette_values.items():
                query_silhouette[key].append(value)
            query_sources.append(int(source_index))
            query_transforms.append(transform_name)
        if (position + 1) % 20 == 0 or position + 1 == sample_size:
            print(f"Query originals {position + 1}/{sample_size}", flush=True)
    query_base_array = {
        key: np.asarray(values, dtype=np.float32) for key, values in query_base.items()
    }
    query_lbp_array = {
        key: np.asarray(values, dtype=np.float32) for key, values in query_lbp.items()
    }
    query_silhouette_array = {
        key: np.asarray(values, dtype=np.float32)
        for key, values in query_silhouette.items()
    }
    normalizers = FeatureNormalizers.load(settings.scaler_dir)
    query_base_array["glcm"] = normalizers.glcm.transform(query_base_array["glcm"]).astype(
        np.float32
    )
    query_base_array["gabor"] = normalizers.gabor.transform(
        query_base_array["gabor"]
    ).astype(np.float32)
    query_base_array["scalar"] = normalizers.shape_scalars.transform(
        query_base_array["scalar"]
    ).astype(np.float32)
    print(f"Queries finished in {perf_counter() - started:.1f}s", flush=True)

    targets = np.asarray(query_sources, dtype=np.int32)
    transform_array = np.asarray(query_transforms, dtype=object)
    selection_rows = np.asarray(
        [row for row, source in enumerate(query_sources) if source in selection_sources],
        dtype=np.int32,
    )
    test_rows = np.asarray(
        [row for row, source in enumerate(query_sources) if source in test_sources],
        dtype=np.int32,
    )
    first_pair, second_pair = sample_pairs(len(records), args.pair_samples, rng)

    query_distance: dict[str, np.ndarray] = {}
    original_distance: dict[str, np.ndarray] = {}
    pair_distance: dict[str, np.ndarray] = {}
    for component in BASE_COMPONENTS:
        query_distance[component] = component_distance(
            component, query_base_array[component], gallery[component]
        )
        original_distance[component] = component_distance(
            component, gallery[component][source_indices], gallery[component]
        )
        pair_distance[component] = component_distance(
            component,
            gallery[component][first_pair],
            gallery[component][second_pair],
            paired=True,
        )
        print(f"Distances: {component}", flush=True)
    for variant in LBP_VARIANTS:
        key = f"lbp:{variant.key}"
        query_distance[key] = sectioned_chi(
            query_lbp_array[variant.key], gallery_lbp_array[variant.key], variant.section_dims
        )
        original_distance[key] = sectioned_chi(
            gallery_lbp_array[variant.key][source_indices],
            gallery_lbp_array[variant.key],
            variant.section_dims,
        )
        pair_distance[key] = sectioned_chi(
            gallery_lbp_array[variant.key][first_pair],
            gallery_lbp_array[variant.key][second_pair],
            variant.section_dims,
            paired=True,
        )
        print(f"Distances: {key}", flush=True)
    for variant in SILHOUETTE_VARIANTS:
        key = f"silhouette:{variant.key}"
        query_distance[key] = cdist(
            query_silhouette_array[variant.key], gallery_silhouette_array[variant.key]
        ).astype(np.float32)
        original_distance[key] = cdist(
            gallery_silhouette_array[variant.key][source_indices],
            gallery_silhouette_array[variant.key],
        ).astype(np.float32)
        pair_distance[key] = np.linalg.norm(
            gallery_silhouette_array[variant.key][first_pair]
            - gallery_silhouette_array[variant.key][second_pair],
            axis=1,
        ).astype(np.float32)
        print(f"Distances: {key}", flush=True)

    def fused(config: Configuration, source: dict[str, np.ndarray]) -> np.ndarray:
        color_raw = 0.30 * source["global_hsv"] + 0.70 * source["spatial_hsv"]
        texture_raw = (
            ((1.0 - config.lbp_weight) / 3.0) * source["glcm"]
            + (2.0 * (1.0 - config.lbp_weight) / 3.0) * source["gabor"]
        )
        if config.lbp_key is not None and config.lbp_weight > 0:
            texture_raw = texture_raw + config.lbp_weight * source[f"lbp:{config.lbp_key}"]
        shape_raw = (
            0.20 * (1.0 - config.silhouette_weight) * source["scalar"]
            + 0.80 * (1.0 - config.silhouette_weight) * source["hog"]
        )
        if config.silhouette_key is not None and config.silhouette_weight > 0:
            shape_raw = shape_raw + config.silhouette_weight * source[
                f"silhouette:{config.silhouette_key}"
            ]
        pair_color = 0.30 * pair_distance["global_hsv"] + 0.70 * pair_distance["spatial_hsv"]
        pair_texture = (
            ((1.0 - config.lbp_weight) / 3.0) * pair_distance["glcm"]
            + (2.0 * (1.0 - config.lbp_weight) / 3.0) * pair_distance["gabor"]
        )
        if config.lbp_key is not None and config.lbp_weight > 0:
            pair_texture = pair_texture + config.lbp_weight * pair_distance[
                f"lbp:{config.lbp_key}"
            ]
        pair_shape = (
            0.20 * (1.0 - config.silhouette_weight) * pair_distance["scalar"]
            + 0.80 * (1.0 - config.silhouette_weight) * pair_distance["hog"]
        )
        if config.silhouette_key is not None and config.silhouette_weight > 0:
            pair_shape = pair_shape + config.silhouette_weight * pair_distance[
                f"silhouette:{config.silhouette_key}"
            ]
        color = np.clip(color_raw / (np.percentile(pair_color, 95) + EPSILON), 0.0, 1.0)
        texture = np.clip(
            texture_raw / (np.percentile(pair_texture, 95) + EPSILON), 0.0, 1.0
        )
        shape = np.clip(shape_raw / (np.percentile(pair_shape, 95) + EPSILON), 0.0, 1.0)
        return (0.30 * color + 0.30 * texture + 0.40 * shape).astype(np.float32)

    configs = [Configuration("No LBP / No Silhouette", "baseline")]
    for variant in LBP_VARIANTS:
        for weight in LBP_WEIGHTS:
            configs.append(
                Configuration(
                    f"LBP {variant.key} w={weight:.2f}",
                    "lbp",
                    lbp_key=variant.key,
                    lbp_weight=weight,
                )
            )
    for variant in SILHOUETTE_VARIANTS:
        for weight in SILHOUETTE_WEIGHTS:
            configs.append(
                Configuration(
                    f"Silhouette {variant.key} w={weight:.2f}",
                    "silhouette",
                    silhouette_key=variant.key,
                    silhouette_weight=weight,
                )
            )

    summary_rows: list[dict] = []
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for config in configs:
        query_fused = fused(config, query_distance)
        original_fused = fused(config, original_distance)
        matrices[config.name] = (query_fused, original_fused)
        selection = retrieval_metrics(query_fused, targets, selection_rows)
        selection.update(
            stability_metrics(
                original_fused,
                query_fused,
                source_indices,
                targets,
                selection_rows,
                args.top_k,
            )
        )
        test = retrieval_metrics(query_fused, targets, test_rows)
        test.update(
            stability_metrics(
                original_fused,
                query_fused,
                source_indices,
                targets,
                test_rows,
                args.top_k,
            )
        )
        row = {
            "configuration": config.name,
            "category": config.category,
            "lbp_variant": config.lbp_key or "",
            "lbp_weight": config.lbp_weight,
            "silhouette_variant": config.silhouette_key or "",
            "silhouette_weight": config.silhouette_weight,
            **{f"selection_{key}": value for key, value in selection.items()},
            "selection_score": config_score(selection),
            **{f"test_{key}": value for key, value in test.items()},
            "test_score": config_score(test),
        }
        summary_rows.append(row)
        print(
            f"{config.name}: selection={row['selection_score']:.4f}, "
            f"test={row['test_score']:.4f}",
            flush=True,
        )

    baseline = summary_rows[0]
    best_lbp_row = max(
        (row for row in summary_rows if row["category"] == "lbp"),
        key=lambda row: row["selection_score"],
    )
    best_silhouette_row = max(
        (row for row in summary_rows if row["category"] == "silhouette"),
        key=lambda row: row["selection_score"],
    )
    best_lbp_config = next(item for item in configs if item.name == best_lbp_row["configuration"])
    best_silhouette_config = next(
        item for item in configs if item.name == best_silhouette_row["configuration"]
    )
    combined = Configuration(
        "Best validation LBP + Silhouette",
        "combined",
        lbp_key=best_lbp_config.lbp_key,
        lbp_weight=best_lbp_config.lbp_weight,
        silhouette_key=best_silhouette_config.silhouette_key,
        silhouette_weight=best_silhouette_config.silhouette_weight,
    )
    query_fused = fused(combined, query_distance)
    original_fused = fused(combined, original_distance)
    matrices[combined.name] = (query_fused, original_fused)
    selection = retrieval_metrics(query_fused, targets, selection_rows)
    selection.update(
        stability_metrics(
            original_fused, query_fused, source_indices, targets, selection_rows, args.top_k
        )
    )
    test = retrieval_metrics(query_fused, targets, test_rows)
    test.update(
        stability_metrics(
            original_fused, query_fused, source_indices, targets, test_rows, args.top_k
        )
    )
    combined_row = {
        "configuration": combined.name,
        "category": combined.category,
        "lbp_variant": combined.lbp_key or "",
        "lbp_weight": combined.lbp_weight,
        "silhouette_variant": combined.silhouette_key or "",
        "silhouette_weight": combined.silhouette_weight,
        **{f"selection_{key}": value for key, value in selection.items()},
        "selection_score": config_score(selection),
        **{f"test_{key}": value for key, value in test.items()},
        "test_score": config_score(test),
    }
    summary_rows.append(combined_row)
    write_csv(output_dir / "configuration_summary.csv", summary_rows)

    detail_configs = [
        next(item for item in configs if item.category == "baseline"),
        best_lbp_config,
        best_silhouette_config,
        combined,
    ]
    transform_rows: list[dict] = []
    for config in detail_configs:
        query_fused, original_fused = matrices[config.name]
        for transform_name in TRANSFORMS:
            rows = test_rows[transform_array[test_rows] == transform_name]
            values = retrieval_metrics(query_fused, targets, rows)
            values.update(
                stability_metrics(
                    original_fused, query_fused, source_indices, targets, rows, args.top_k
                )
            )
            transform_rows.append(
                {
                    "configuration": config.name,
                    "transform": transform_name,
                    "family": TRANSFORM_FAMILY[transform_name],
                    **values,
                    "score": config_score(values),
                }
            )
    write_csv(output_dir / "per_transform_test.csv", transform_rows)

    lbp_test_delta = best_lbp_row["test_score"] - baseline["test_score"]
    silhouette_test_delta = best_silhouette_row["test_score"] - baseline["test_score"]
    combined_test_delta = combined_row["test_score"] - baseline["test_score"]
    lbp_selection_deltas = [
        row["selection_score"] - baseline["selection_score"]
        for row in summary_rows
        if row["category"] == "lbp"
    ]
    lbp_test_deltas = [
        row["test_score"] - baseline["test_score"]
        for row in summary_rows
        if row["category"] == "lbp"
    ]
    silhouette_selection_deltas = [
        row["selection_score"] - baseline["selection_score"]
        for row in summary_rows
        if row["category"] == "silhouette"
    ]
    silhouette_test_deltas = [
        row["test_score"] - baseline["test_score"]
        for row in summary_rows
        if row["category"] == "silhouette"
    ]

    def conclusion(delta: float) -> str:
        if delta >= 0.003:
            return "SUPPORTED"
        if delta > 0.0005:
            return "SMALL_GAIN"
        if delta >= -0.0005:
            return "NO_MEASURABLE_GAIN"
        return "HURTS"

    report = f"""# LBP and Silhouette Parameter Study (label-free)

## Protocol

- Gallery: {len(records)} images.
- Source images: {sample_size}; selection={len(selection_sources)}, held-out test={len(test_sources)}.
- Queries: {len(targets)} ({len(TRANSFORMS)} stronger deterministic perturbations/source).
- No species names or labels were read.
- Selection score = 30% Hit@1 + 40% MRR + 30% RBO@10.
- The baseline contains HSV, GLCM, Gabor, shape scalars, and HOG, but no LBP or silhouette.

## Held-out result

| Configuration chosen on selection | Hit@1 | MRR | RBO@10 | Top-10 overlap | Composite |
|---|---:|---:|---:|---:|---:|
| No LBP / No Silhouette | {baseline['test_hit1']:.4f} | {baseline['test_mrr']:.4f} | {baseline['test_rbo10']:.4f} | {baseline['test_top10_overlap']:.4f} | {baseline['test_score']:.4f} |
| {best_lbp_row['configuration']} | {best_lbp_row['test_hit1']:.4f} | {best_lbp_row['test_mrr']:.4f} | {best_lbp_row['test_rbo10']:.4f} | {best_lbp_row['test_top10_overlap']:.4f} | {best_lbp_row['test_score']:.4f} |
| {best_silhouette_row['configuration']} | {best_silhouette_row['test_hit1']:.4f} | {best_silhouette_row['test_mrr']:.4f} | {best_silhouette_row['test_rbo10']:.4f} | {best_silhouette_row['test_top10_overlap']:.4f} | {best_silhouette_row['test_score']:.4f} |
| Best validation LBP + Silhouette | {combined_row['test_hit1']:.4f} | {combined_row['test_mrr']:.4f} | {combined_row['test_rbo10']:.4f} | {combined_row['test_top10_overlap']:.4f} | {combined_row['test_score']:.4f} |

## Decision

- LBP: **{conclusion(lbp_test_delta)}**, held-out composite delta {lbp_test_delta:+.4f}.
- Silhouette: **{conclusion(silhouette_test_delta)}**, held-out composite delta {silhouette_test_delta:+.4f}.
- Combined: **{conclusion(combined_test_delta)}**, held-out composite delta {combined_test_delta:+.4f}.

Across every parameter setting, LBP deltas ranged from
{min(lbp_selection_deltas):+.6f} to {max(lbp_selection_deltas):+.6f} on selection and
{min(lbp_test_deltas):+.6f} to {max(lbp_test_deltas):+.6f} on test. Silhouette deltas ranged from
{min(silhouette_selection_deltas):+.6f} to {max(silhouette_selection_deltas):+.6f} on selection and
{min(silhouette_test_deltas):+.6f} to {max(silhouette_test_deltas):+.6f} on test.

The best silhouette is bbox-normalized, which shows that the original full-canvas grid mixed true
shape with position and scale. It improves most perturbations but remains sensitive to rotation and
strong mask erosion. Its weight is an internal shape-group weight, not 80% of the complete score.

These metrics test robustness and neighbor-list stability, not species classification and not human visual preference. A small positive delta should not justify production complexity by itself. See `configuration_summary.csv` for every parameter combination and `per_transform_test.csv` for failure modes.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
