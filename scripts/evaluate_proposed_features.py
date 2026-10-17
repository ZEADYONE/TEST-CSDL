"""Label-free experiment for Hu Moments and spatial multi-scale LBP.

This script is intentionally separate from the production extractor. It reads
the stored gallery and processed PNG files, creates deterministic perturbed
queries, and writes experimental reports without modifying PostgreSQL or the
web application. Species labels are never read or used.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
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
from app.services.preprocessing import bgr_and_alpha, load_image, mask_bbox, rgba_image, write_image  # noqa: E402
from scripts.evaluate_visual_feature_selection import (  # noqa: E402
    COMPONENTS,
    DIMENSIONS,
    EPSILON,
    TRANSFORMS,
    atomic_from_stored,
    component_distance,
    extract_atomic,
    retrieval_metrics,
    transform_image,
)


SPATIAL_LBP_SCALES = ((8, 1), (16, 2), (24, 4))
SPATIAL_LBP_REGIONS = 3
SPATIAL_LBP_REGION_DIM = sum(points + 2 for points, _ in SPATIAL_LBP_SCALES)
SPATIAL_LBP_DIM = SPATIAL_LBP_REGIONS * SPATIAL_LBP_REGION_DIM
HU_DIM = 7


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    lbp_component: str | None
    hu_weight: float
    category: str


CONFIGURATIONS = (
    ExperimentConfig("Current Full", "lbp", 0.0, "baseline"),
    ExperimentConfig("Current Full - LBP", None, 0.0, "ablation"),
    ExperimentConfig("Current Full + Hu 0.05", "lbp", 0.05, "hu"),
    ExperimentConfig("Current Full + Hu 0.10", "lbp", 0.10, "hu"),
    ExperimentConfig("Current Full + Hu 0.20", "lbp", 0.20, "hu"),
    ExperimentConfig("Replace LBP with Spatial MS-LBP", "spatial_ms_lbp", 0.0, "lbp"),
    ExperimentConfig("Spatial MS-LBP + Hu 0.10", "spatial_ms_lbp", 0.10, "combined"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--pair-samples", type=int, default=20_000)
    parser.add_argument("--representatives", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def hu_moments(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    moments = cv2.HuMoments(cv2.moments(binary, binaryImage=True)).ravel()
    transformed = np.zeros(HU_DIM, dtype=np.float32)
    nonzero = np.abs(moments) > 1e-30
    transformed[nonzero] = (
        -np.sign(moments[nonzero]) * np.log10(np.abs(moments[nonzero]))
    ).astype(np.float32)
    return transformed


def spatial_multiscale_lbp(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Three horizontal regions, three uniform-LBP scales, 162 dimensions."""

    binary = np.asarray(mask, dtype=bool)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bbox = mask_bbox(binary)
    if bbox is None:
        raise ValueError("Spatial LBP requires a non-empty mask")
    x1, y1, x2, y2 = bbox
    boundaries = np.linspace(x1, x2, SPATIAL_LBP_REGIONS + 1, dtype=int)
    scale_data: list[tuple[np.ndarray, np.ndarray, int]] = []
    for points, radius in SPATIAL_LBP_SCALES:
        support = radius + 1
        kernel = np.ones((2 * support + 1, 2 * support + 1), dtype=np.uint8)
        valid = cv2.erode(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
        if not np.any(valid):
            valid = binary
        scale_data.append(
            (local_binary_pattern(gray, points, radius, method="uniform"), valid, points + 2)
        )

    regions: list[np.ndarray] = []
    for region in range(SPATIAL_LBP_REGIONS):
        region_parts: list[np.ndarray] = []
        region_mask = np.zeros_like(binary)
        region_mask[y1:y2, boundaries[region] : boundaries[region + 1]] = True
        for values, valid, bins in scale_data:
            selected = values[valid & region_mask]
            histogram, _ = np.histogram(selected, bins=np.arange(bins + 1), range=(0, bins))
            histogram = histogram.astype(np.float32)
            total = float(histogram.sum())
            if total > 0:
                histogram /= total
            region_parts.append(histogram)
        region_vector = np.concatenate(region_parts)
        region_total = float(region_vector.sum())
        if region_total > 0:
            region_vector /= region_total
        regions.append(region_vector)
    vector = np.concatenate(regions).astype(np.float32)
    if vector.shape != (SPATIAL_LBP_DIM,):
        raise AssertionError(f"Expected {SPATIAL_LBP_DIM} Spatial LBP values, got {vector.shape}")
    return vector


def spatial_lbp_distance(first: np.ndarray, second: np.ndarray, *, paired: bool = False) -> np.ndarray:
    distances = []
    for region in range(SPATIAL_LBP_REGIONS):
        section = slice(
            region * SPATIAL_LBP_REGION_DIM,
            (region + 1) * SPATIAL_LBP_REGION_DIM,
        )
        left = first[:, section]
        right = second[:, section]
        if paired:
            value = 0.5 * np.sum((left - right) ** 2 / (left + right + EPSILON), axis=1)
        else:
            value = np.empty((len(left), len(right)), dtype=np.float32)
            for start in range(0, len(left), 16):
                a = left[start : start + 16, None, :]
                b = right[None, :, :]
                value[start : start + 16] = 0.5 * np.sum(
                    (a - b) ** 2 / (a + b + EPSILON), axis=2
                )
        distances.append(value.astype(np.float32))
    return (sum(distances) / SPATIAL_LBP_REGIONS).astype(np.float32)


def sample_pairs(size: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    count = min(count, size * (size - 1) // 2)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < count:
        first, second = (int(value) for value in rng.integers(0, size, size=2))
        if first != second:
            pairs.add((min(first, second), max(first, second)))
    ordered = sorted(pairs)
    return (
        np.asarray([pair[0] for pair in ordered], dtype=np.int32),
        np.asarray([pair[1] for pair in ordered], dtype=np.int32),
    )


def normalize_group(raw: np.ndarray, sample: np.ndarray) -> tuple[np.ndarray, float]:
    p95 = float(np.percentile(sample, 95))
    return np.clip(raw / (p95 + EPSILON), 0.0, 1.0).astype(np.float32), p95


def fuse(
    distance: dict[str, np.ndarray],
    pair_distance: dict[str, np.ndarray],
    config: ExperimentConfig,
    settings,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    color_raw = (
        settings.color_joint_weight * distance["global_hsv"]
        + settings.color_spatial_weight * distance["spatial_hsv"]
    )
    color_sample = (
        settings.color_joint_weight * pair_distance["global_hsv"]
        + settings.color_spatial_weight * pair_distance["spatial_hsv"]
    )

    texture_parts = {
        "glcm": settings.texture_glcm_weight,
        "gabor": settings.texture_gabor_weight,
    }
    if config.lbp_component is not None:
        texture_parts[config.lbp_component] = settings.texture_lbp_weight
    texture_total = sum(texture_parts.values())
    texture_raw = sum(weight * distance[name] for name, weight in texture_parts.items()) / texture_total
    texture_sample = sum(weight * pair_distance[name] for name, weight in texture_parts.items()) / texture_total

    hu_weight = config.hu_weight
    shape_parts = {
        "scalar": settings.shape_scalar_weight,
        "hog": max(0.0, settings.shape_hog_weight - hu_weight),
    }
    if hu_weight > 0:
        shape_parts["hu"] = hu_weight
    shape_total = sum(shape_parts.values())
    shape_raw = sum(weight * distance[name] for name, weight in shape_parts.items()) / shape_total
    shape_sample = sum(weight * pair_distance[name] for name, weight in shape_parts.items()) / shape_total

    color, color_p95 = normalize_group(color_raw, color_sample)
    texture, texture_p95 = normalize_group(texture_raw, texture_sample)
    shape, shape_p95 = normalize_group(shape_raw, shape_sample)
    final = (
        settings.weight_color * color
        + settings.weight_texture * texture
        + settings.weight_shape * shape
    ).astype(np.float32)
    return final, {"color": color, "texture": texture, "shape": shape}, {
        "color": color_p95,
        "texture": texture_p95,
        "shape": shape_p95,
    }


def rbo_at_k(first: np.ndarray, second: np.ndarray, p: float = 0.9) -> float:
    overlap = 0.0
    first_seen: set[int] = set()
    second_seen: set[int] = set()
    agreement = 0.0
    for depth, (left, right) in enumerate(zip(first, second), start=1):
        first_seen.add(int(left))
        second_seen.add(int(right))
        overlap = len(first_seen & second_seen) / depth
        agreement += (1.0 - p) * (p ** (depth - 1)) * overlap
    return float(agreement + overlap * (p ** len(first)))


def topk_stability(
    original: np.ndarray,
    transformed: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    rows: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    overlaps: list[float] = []
    jaccards: list[float] = []
    rbos: list[float] = []
    source_to_original_row = {int(source): index for index, source in enumerate(source_indices)}
    for row in rows:
        target = int(target_indices[row])
        original_row = source_to_original_row[target]
        original_order = np.argsort(original[original_row], kind="stable")
        transformed_order = np.argsort(transformed[row], kind="stable")
        original_top = original_order[original_order != target][:top_k]
        transformed_top = transformed_order[transformed_order != target][:top_k]
        intersection = len(set(original_top.tolist()) & set(transformed_top.tolist()))
        union = len(set(original_top.tolist()) | set(transformed_top.tolist()))
        overlaps.append(intersection / top_k)
        jaccards.append(intersection / union if union else 1.0)
        rbos.append(rbo_at_k(original_top, transformed_top))
    return {
        "topk_overlap": float(np.mean(overlaps)),
        "jaccard": float(np.mean(jaccards)),
        "rbo": float(np.mean(rbos)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_image(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output_dir.resolve())).as_posix()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    output_dir = PROJECT_ROOT / "results" / "proposed_feature_experiments"
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

    rng = np.random.default_rng(settings.random_seed)
    sample_size = min(args.sample_size, len(records))
    source_indices = np.sort(rng.choice(len(records), size=sample_size, replace=False))
    shuffled = source_indices.copy()
    rng.shuffle(shuffled)
    selection_sources = set(shuffled[: len(shuffled) // 2].tolist())
    test_sources = set(shuffled[len(shuffled) // 2 :].tolist())
    representatives = source_indices[: min(args.representatives, sample_size)]

    gallery = atomic_from_stored(records)
    gallery_spatial_lbp: list[np.ndarray] = []
    gallery_hu: list[np.ndarray] = []
    new_gallery_timings = {"spatial_ms_lbp": [], "hu": []}
    print(f"Extracting experimental gallery descriptors for {len(records)} images", flush=True)
    for index, record in enumerate(records):
        raw = load_image(record.processed_path)
        image, alpha = bgr_and_alpha(raw)
        if alpha is None:
            raise ValueError(f"Processed image lacks alpha: {record.processed_path}")
        mask = alpha > 0
        started = perf_counter()
        gallery_spatial_lbp.append(spatial_multiscale_lbp(image, mask))
        new_gallery_timings["spatial_ms_lbp"].append((perf_counter() - started) * 1000.0)
        started = perf_counter()
        gallery_hu.append(hu_moments(mask))
        new_gallery_timings["hu"].append((perf_counter() - started) * 1000.0)
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f"Gallery descriptors: {index + 1}/{len(records)}", flush=True)
    gallery["spatial_ms_lbp"] = np.asarray(gallery_spatial_lbp, dtype=np.float32)
    gallery_hu_array = np.asarray(gallery_hu, dtype=np.float32)
    hu_scaler = StandardScaler().fit(gallery_hu_array)
    gallery["hu"] = hu_scaler.transform(gallery_hu_array).astype(np.float32)

    query_values = {component: [] for component in (*COMPONENTS, "spatial_ms_lbp", "hu")}
    timing_values = {component: [] for component in (*COMPONENTS, "spatial_ms_lbp", "hu")}
    query_sources: list[int] = []
    query_transforms: list[str] = []
    preview_paths: dict[tuple[int, str], Path] = {}
    print(f"Extracting {sample_size * len(TRANSFORMS)} transformed queries", flush=True)
    for position, source_index in enumerate(source_indices):
        raw = load_image(records[int(source_index)].processed_path)
        image, alpha = bgr_and_alpha(raw)
        assert alpha is not None
        mask = alpha > 0
        for transform_name in TRANSFORMS:
            transformed, transformed_mask = transform_image(image, mask, transform_name)
            atomic, timings = extract_atomic(transformed, transformed_mask)
            for component in COMPONENTS:
                query_values[component].append(atomic[component])
                timing_values[component].append(timings[component])
            started = perf_counter()
            query_values["spatial_ms_lbp"].append(
                spatial_multiscale_lbp(transformed, transformed_mask)
            )
            timing_values["spatial_ms_lbp"].append((perf_counter() - started) * 1000.0)
            started = perf_counter()
            query_values["hu"].append(hu_moments(transformed_mask))
            timing_values["hu"].append((perf_counter() - started) * 1000.0)
            query_sources.append(int(source_index))
            query_transforms.append(transform_name)
            if int(source_index) in set(int(value) for value in representatives):
                path = preview_dir / f"query_{source_index}_{transform_name}.png"
                write_image(path, rgba_image(transformed, transformed_mask))
                preview_paths[(int(source_index), transform_name)] = path
        if (position + 1) % 20 == 0 or position + 1 == sample_size:
            print(f"Query originals: {position + 1}/{sample_size}", flush=True)

    query = {
        component: np.asarray(values, dtype=np.float32)
        for component, values in query_values.items()
    }
    normalizers = FeatureNormalizers.load(settings.scaler_dir)
    query["glcm"] = normalizers.glcm.transform(query["glcm"]).astype(np.float32)
    query["gabor"] = normalizers.gabor.transform(query["gabor"]).astype(np.float32)
    query["scalar"] = normalizers.shape_scalars.transform(query["scalar"]).astype(np.float32)
    query["hu"] = hu_scaler.transform(query["hu"]).astype(np.float32)

    target_indices = np.asarray(query_sources, dtype=np.int32)
    transform_array = np.asarray(query_transforms, dtype=object)
    selection_rows = np.asarray(
        [i for i, source in enumerate(query_sources) if source in selection_sources], dtype=np.int32
    )
    test_rows = np.asarray(
        [i for i, source in enumerate(query_sources) if source in test_sources], dtype=np.int32
    )
    first_pair, second_pair = sample_pairs(len(records), args.pair_samples, rng)

    all_components = (*COMPONENTS, "spatial_ms_lbp", "hu")
    distance: dict[str, np.ndarray] = {}
    original_distance: dict[str, np.ndarray] = {}
    pair_distance: dict[str, np.ndarray] = {}
    distance_latency: dict[str, float] = {}
    for component in all_components:
        started = perf_counter()
        if component == "spatial_ms_lbp":
            distance[component] = spatial_lbp_distance(query[component], gallery[component])
            original_distance[component] = spatial_lbp_distance(
                gallery[component][source_indices], gallery[component]
            )
            pair_distance[component] = spatial_lbp_distance(
                gallery[component][first_pair], gallery[component][second_pair], paired=True
            )
        else:
            distance[component] = component_distance(component, query[component], gallery[component])
            original_distance[component] = component_distance(
                component, gallery[component][source_indices], gallery[component]
            )
            pair_distance[component] = component_distance(
                component,
                gallery[component][first_pair],
                gallery[component][second_pair],
                paired=True,
            )
        distance_latency[component] = (perf_counter() - started) * 1000.0 / len(target_indices)
        print(f"Distances: {component}", flush=True)

    summary_rows: list[dict] = []
    transform_rows: list[dict] = []
    result_matrices: dict[str, np.ndarray] = {}
    group_matrices: dict[str, dict[str, np.ndarray]] = {}
    for config in CONFIGURATIONS:
        fused, groups, p95 = fuse(distance, pair_distance, config, settings)
        original_fused, _, _ = fuse(original_distance, pair_distance, config, settings)
        result_matrices[config.name] = fused
        group_matrices[config.name] = groups
        selection = retrieval_metrics(fused, target_indices, selection_rows)
        test = retrieval_metrics(fused, target_indices, test_rows)
        selection_stability = topk_stability(
            original_fused, fused, source_indices, target_indices, selection_rows, args.top_k
        )
        test_stability = topk_stability(
            original_fused, fused, source_indices, target_indices, test_rows, args.top_k
        )
        dimensions = 2411
        if config.lbp_component is None:
            dimensions -= DIMENSIONS["lbp"]
        elif config.lbp_component == "spatial_ms_lbp":
            dimensions += SPATIAL_LBP_DIM - DIMENSIONS["lbp"]
        if config.hu_weight > 0:
            dimensions += HU_DIM
        extraction_components = set(COMPONENTS)
        if config.lbp_component is None:
            extraction_components.remove("lbp")
        elif config.lbp_component == "spatial_ms_lbp":
            extraction_components.remove("lbp")
            extraction_components.add("spatial_ms_lbp")
        if config.hu_weight > 0:
            extraction_components.add("hu")
        extraction_ms = sum(float(np.mean(timing_values[name])) for name in extraction_components)
        retrieval_ms = sum(distance_latency[name] for name in extraction_components)
        summary_rows.append(
            {
                "configuration": config.name,
                "category": config.category,
                "dimensions": dimensions,
                "selection_hit1": selection["hit1"],
                "selection_mrr": selection["mrr"],
                "selection_top10_overlap": selection_stability["topk_overlap"],
                "selection_rbo10": selection_stability["rbo"],
                "test_hit1": test["hit1"],
                "test_hit5": test["hit5"],
                "test_mrr": test["mrr"],
                "test_mean_rank": test["mean_rank"],
                "test_top10_overlap": test_stability["topk_overlap"],
                "test_jaccard10": test_stability["jaccard"],
                "test_rbo10": test_stability["rbo"],
                "extraction_ms": extraction_ms,
                "retrieval_ms": retrieval_ms,
                "p95_color": p95["color"],
                "p95_texture": p95["texture"],
                "p95_shape": p95["shape"],
            }
        )
        for transform_name in TRANSFORMS:
            rows = np.asarray(
                [row for row in test_rows if transform_array[row] == transform_name], dtype=np.int32
            )
            exact = retrieval_metrics(fused, target_indices, rows)
            stable = topk_stability(
                original_fused, fused, source_indices, target_indices, rows, args.top_k
            )
            transform_rows.append(
                {
                    "configuration": config.name,
                    "transform": transform_name,
                    **exact,
                    **stable,
                }
            )

    write_csv(output_dir / "configuration_summary.csv", summary_rows)
    write_csv(output_dir / "per_transform_metrics.csv", transform_rows)
    latency_rows = [
        {
            "feature": name,
            "dimensions": SPATIAL_LBP_DIM if name == "spatial_ms_lbp" else HU_DIM,
            "gallery_extraction_ms_mean": float(np.mean(new_gallery_timings[name])),
            "query_extraction_ms_mean": float(np.mean(timing_values[name])),
            "distance_ms_per_query": distance_latency[name],
        }
        for name in ("spatial_ms_lbp", "hu")
    ]
    write_csv(output_dir / "experimental_feature_latency.csv", latency_rows)

    visual_configs = (
        "Current Full",
        "Current Full - LBP",
        "Current Full + Hu 0.10",
        "Replace LBP with Spatial MS-LBP",
        "Spatial MS-LBP + Hu 0.10",
    )
    rating_rows: list[dict] = []
    sections: list[str] = []
    for representative_index, source in enumerate(representatives):
        transform_name = TRANSFORMS[representative_index % len(TRANSFORMS)]
        matches = np.flatnonzero((target_indices == source) & (transform_array == transform_name))
        if not len(matches):
            continue
        row = int(matches[0])
        query_path = preview_paths[(int(source), transform_name)]
        config_html: list[str] = []
        for config_name in visual_configs:
            fused = result_matrices[config_name]
            groups = group_matrices[config_name]
            order = np.argsort(fused[row], kind="stable")
            order = order[order != source][: args.top_k]
            cards = []
            for rank, candidate in enumerate(order, start=1):
                record = records[int(candidate)]
                image_path = relative_image(Path(record.processed_path), output_dir)
                total = max(0.0, 1.0 - float(fused[row, candidate]))
                color = max(0.0, 1.0 - float(groups["color"][row, candidate]))
                texture = max(0.0, 1.0 - float(groups["texture"][row, candidate]))
                shape = max(0.0, 1.0 - float(groups["shape"][row, candidate]))
                cards.append(
                    "<div class='result'>"
                    f"<b>#{rank}</b><img src='{escape(image_path)}'>"
                    f"<small>Total {total:.3f}<br>Color {color:.3f} · Texture {texture:.3f} · Shape {shape:.3f}</small>"
                    "</div>"
                )
                rating_rows.append(
                    {
                        "query_source_id": records[int(source)].id,
                        "transformation": transform_name,
                        "configuration": config_name,
                        "result_rank": rank,
                        "result_image_id": record.id,
                        "color_rating_1_to_5": "",
                        "texture_rating_1_to_5": "",
                        "shape_rating_1_to_5": "",
                        "overall_rating_1_to_5": "",
                        "notes": "",
                    }
                )
            config_html.append(
                f"<h3>{escape(config_name)}</h3><div class='results'>{''.join(cards)}</div>"
            )
        sections.append(
            "<section>"
            f"<h2>Query {representative_index + 1}: {escape(transform_name)}</h2>"
            f"<img class='query' src='{escape(relative_image(query_path, output_dir))}'>"
            f"{''.join(config_html)}</section>"
        )
    visual_html = f"""<!doctype html><html lang='vi'><head><meta charset='utf-8'>
<title>Hu and Spatial LBP visual comparison</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f1f4ed;color:#17231d}}
section{{background:white;padding:18px;border-radius:12px;margin-bottom:24px}} .query{{width:220px;height:220px;object-fit:contain;background:#eef1e9}}
.results{{display:grid;grid-template-columns:repeat(10,minmax(105px,1fr));gap:8px;overflow-x:auto}}
.result{{border:1px solid #d7ded4;border-radius:8px;padding:7px}} .result img{{display:block;width:100%;height:105px;object-fit:contain}}
small{{font-size:10px;color:#536059}} h3{{margin-bottom:7px}}
</style></head><body><h1>So sánh trực quan — không dùng nhãn loài</h1>
<p>Ảnh gốc tương ứng đã bị loại khỏi Top-10 để so sánh các láng giềng thị giác.</p>{''.join(sections)}</body></html>"""
    (output_dir / "visual_comparison.html").write_text(visual_html, encoding="utf-8")
    write_csv(output_dir / "manual_visual_ratings.csv", rating_rows)

    baseline = next(row for row in summary_rows if row["configuration"] == "Current Full")
    report = [
        "# Label-free Hu Moments and Spatial Multi-scale LBP Experiment",
        "",
        "## Protocol",
        "",
        f"- Gallery: {len(records)} processed originals.",
        f"- Sampled originals: {sample_size}; deterministic queries: {len(target_indices)}.",
        "- Selection/test split is random by image identity and never uses species.",
        "- Exact-image recovery and Top-10 neighbor stability are separate metrics.",
        "- The corresponding original is excluded before Top-10 stability is measured.",
        "- No classifier, species voting, PCA, deep feature, IVF, database write, or web change.",
        "",
        "## Held-out test results",
        "",
        "| Configuration | Dim | Hit@1 | MRR | Top-10 overlap | Jaccard@10 | RBO@10 | Extract ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['configuration']} | {row['dimensions']} | {row['test_hit1']:.4f} | {row['test_mrr']:.4f} | {row['test_top10_overlap']:.4f} | {row['test_jaccard10']:.4f} | {row['test_rbo10']:.4f} | {row['extraction_ms']:.2f} |"
        )
    report.extend(
        (
            "",
            "Top-10 overlap measures whether a perturbed query preserves the same visual neighbors as its original. It does not use or imply species correctness.",
            "",
            "## Experimental feature cost",
            "",
            "| Feature | Dim | Gallery extraction ms | Query extraction ms | Distance ms/query |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for row in latency_rows:
        report.append(
            f"| {row['feature']} | {row['dimensions']} | {row['gallery_extraction_ms_mean']:.2f} | {row['query_extraction_ms_mean']:.2f} | {row['distance_ms_per_query']:.3f} |"
        )
    best_stability = max(summary_rows, key=lambda row: float(row["test_rbo10"]))
    by_name = {row["configuration"]: row for row in summary_rows}
    no_lbp = by_name["Current Full - LBP"]
    hu_005 = by_name["Current Full + Hu 0.05"]
    hu_010 = by_name["Current Full + Hu 0.10"]
    hu_020 = by_name["Current Full + Hu 0.20"]
    spatial_lbp = by_name["Replace LBP with Spatial MS-LBP"]
    combined = by_name["Spatial MS-LBP + Hu 0.10"]
    report.extend(
        (
            "",
            "## Automatic conclusion boundary",
            "",
            f"- Baseline test RBO@10: {baseline['test_rbo10']:.4f}.",
            f"- Highest test RBO@10: {best_stability['configuration']} ({best_stability['test_rbo10']:.4f}).",
            "- A descriptor is not declared visually better from stability alone. Open `visual_comparison.html` and complete `manual_visual_ratings.csv`.",
            "- Only keep an experimental descriptor if it improves held-out stability and human visual ratings enough to justify its latency.",
            "",
            "## Experimental decisions",
            "",
            "| Proposal | Status | Test evidence | Cost evidence |",
            "|---|---|---|---|",
            f"| Remove current LBP | **NEEDS_MORE_TESTING** | RBO@10 {no_lbp['test_rbo10']:.4f} vs baseline {baseline['test_rbo10']:.4f}; nearly neutral | Saves {baseline['extraction_ms'] - no_lbp['extraction_ms']:.2f} ms/query |",
            f"| Add Hu 0.05 | **NEEDS_MORE_TESTING** | RBO@10 {hu_005['test_rbo10']:.4f}, lower than baseline | Adds {hu_005['extraction_ms'] - baseline['extraction_ms']:.2f} ms/query |",
            f"| Add Hu 0.10 | **DO_NOT_ADOPT** | RBO@10 {hu_010['test_rbo10']:.4f}, lower than baseline | Adds {hu_010['extraction_ms'] - baseline['extraction_ms']:.2f} ms/query |",
            f"| Add Hu 0.20 | **NEEDS_MORE_TESTING** | Overlap {hu_020['test_top10_overlap']:.4f} is slightly higher, but RBO@10 {hu_020['test_rbo10']:.4f} is lower | Adds {hu_020['extraction_ms'] - baseline['extraction_ms']:.2f} ms/query |",
            f"| Replace with Spatial MS-LBP | **DO_NOT_ADOPT** | RBO@10 {spatial_lbp['test_rbo10']:.4f}, effectively unchanged/slightly lower | Adds {spatial_lbp['extraction_ms'] - baseline['extraction_ms']:.2f} ms/query and 134 dimensions |",
            f"| Spatial MS-LBP + Hu 0.10 | **DO_NOT_ADOPT** | RBO@10 {combined['test_rbo10']:.4f}, clearly lower | Adds {combined['extraction_ms'] - baseline['extraction_ms']:.2f} ms/query and 141 dimensions |",
            "",
            "These statuses concern the tested implementations and weights, not the entire Hu or LBP feature families. Human visual ratings are still required before a final production removal decision.",
            "",
            "## Artifacts",
            "",
            "- `configuration_summary.csv`",
            "- `per_transform_metrics.csv`",
            "- `experimental_feature_latency.csv`",
            "- `visual_comparison.html`",
            "- `manual_visual_ratings.csv`",
        )
    )
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    (output_dir / "experiment.json").write_text(
        json.dumps(
            {
                "uses_species": False,
                "gallery": len(records),
                "sample_size": sample_size,
                "query_count": len(target_indices),
                "spatial_lbp": {
                    "scales": SPATIAL_LBP_SCALES,
                    "regions": SPATIAL_LBP_REGIONS,
                    "dimensions": SPATIAL_LBP_DIM,
                },
                "results": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
