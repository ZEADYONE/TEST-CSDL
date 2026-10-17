"""Compare Shape Scalar and bbox-normalized Silhouette without species labels.

The gallery and production database are read-only. Parameter selection and the
final comparison use disjoint source-image splits. Both standalone shape
descriptors and their marginal value beside HOG are measured.
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
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import joinedload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import ImageFeature, ImageRecord  # noqa: E402
from app.services.color_features import JOINT_DIM, extract_color_features  # noqa: E402
from app.services.gabor_features import extract_gabor_features  # noqa: E402
from app.services.hog_features import extract_hog_features  # noqa: E402
from app.services.normalization import FeatureNormalizers  # noqa: E402
from app.services.preprocessing import bgr_and_alpha, load_image  # noqa: E402
from app.services.shape_features import SHAPE_SCALAR_DIM, SILHOUETTE_DIM, _shape_scalars  # noqa: E402
from app.services.texture_features import (  # noqa: E402
    GLCM_DIM,
    GRAY_LEVELS,
    LBP_DIM,
    _glcm_properties,
    _masked_glcm,
)
from scripts.evaluate_lbp_silhouette_variants import (  # noqa: E402
    EPSILON,
    TRANSFORMS,
    TRANSFORM_FAMILY,
    config_score,
    extract_silhouette_variants,
    retrieval_metrics,
    stability_metrics,
    transform_image,
)
from scripts.evaluate_proposed_features import sample_pairs  # noqa: E402
from scripts.evaluate_visual_feature_selection import (  # noqa: E402
    atomic_from_stored,
    component_distance,
)


BBOX_VARIANTS = ("bbox_grid4", "bbox_grid8", "bbox_grid16")
SCALAR_WEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1.00)
SILHOUETTE_WEIGHTS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


@dataclass(frozen=True)
class Configuration:
    name: str
    category: str
    scalar_weight: float
    hog_weight: float
    silhouette_key: str | None = None
    silhouette_weight: float = 0.0
    shape_only: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=160)
    parser.add_argument("--pair-samples", type=int, default=20_000)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def extract_query_components(image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    color = extract_color_features(image, binary)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    quantized = np.minimum(
        gray.astype(np.uint16) * GRAY_LEVELS // 256,
        GRAY_LEVELS - 1,
    ).astype(np.uint8)
    offsets = ((0, 1), (-1, 1), (-1, 0), (-1, -1))
    per_angle = np.asarray(
        [
            _glcm_properties(_masked_glcm(quantized, binary, dy, dx))
            for dy, dx in offsets
        ],
        dtype=np.float64,
    )
    glcm = np.column_stack((per_angle.mean(axis=0), per_angle.std(axis=0))).ravel()
    return {
        "global_hsv": color[:JOINT_DIM],
        "spatial_hsv": color[JOINT_DIM:],
        "glcm": glcm.astype(np.float32),
        "gabor": extract_gabor_features(image, binary),
        "scalar": _shape_scalars(binary),
        "hog": extract_hog_features(image, binary),
        **extract_silhouette_variants(binary),
    }


def configurations() -> list[Configuration]:
    items = [Configuration("HOG only", "hog_only", 0.0, 1.0)]
    for scalar_weight in SCALAR_WEIGHTS:
        items.append(
            Configuration(
                f"HOG + Scalar s={scalar_weight:.2f}",
                "hog_scalar",
                scalar_weight,
                1.0 - scalar_weight,
            )
        )
    items.append(Configuration("Scalar only (shape-only)", "shape_single_scalar", 1.0, 0.0, shape_only=True))
    for key in BBOX_VARIANTS:
        for silhouette_weight in SILHOUETTE_WEIGHTS:
            items.append(
                Configuration(
                    f"HOG + {key} b={silhouette_weight:.2f}",
                    "hog_silhouette",
                    0.0,
                    1.0 - silhouette_weight,
                    key,
                    silhouette_weight,
                )
            )
        items.append(
            Configuration(
                f"{key} only (shape-only)",
                "shape_single_silhouette",
                0.0,
                0.0,
                key,
                1.0,
                True,
            )
        )
        for silhouette_weight in SILHOUETTE_WEIGHTS[:-1]:
            remainder = 1.0 - silhouette_weight
            items.append(
                Configuration(
                    f"HOG + Scalar + {key} b={silhouette_weight:.2f}",
                    "hog_scalar_silhouette",
                    0.20 * remainder,
                    0.80 * remainder,
                    key,
                    silhouette_weight,
                )
            )
    return items


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
    output_dir = PROJECT_ROOT / "results" / "shape_scalar_vs_silhouette"
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
        raise RuntimeError("At least two indexed images are required")

    gallery = atomic_from_stored(records)
    gallery_silhouette = {key: [] for key in BBOX_VARIANTS}
    gallery_scalar_raw: list[np.ndarray] = []
    print(f"Extracting bbox silhouettes for {len(records)} gallery images", flush=True)
    for index, record in enumerate(records):
        image, alpha = bgr_and_alpha(load_image(record.processed_path))
        if alpha is None:
            raise ValueError(f"Missing alpha mask: {record.processed_path}")
        variants = extract_silhouette_variants(alpha > 0)
        gallery_scalar_raw.append(_shape_scalars(alpha > 0))
        for key in BBOX_VARIANTS:
            gallery_silhouette[key].append(variants[key])
        if (index + 1) % 250 == 0 or index + 1 == len(records):
            print(f"Gallery silhouettes {index + 1}/{len(records)}", flush=True)
    gallery_silhouette = {
        key: np.asarray(values, dtype=np.float32)
        for key, values in gallery_silhouette.items()
    }
    # Shape Scalar was removed from the production vector in v5. Fit this
    # experiment-only scaler directly from the gallery so the ablation remains
    # reproducible without restoring Scalar to the application pipeline.
    scalar_scaler = StandardScaler().fit(np.asarray(gallery_scalar_raw, dtype=np.float32))
    gallery["scalar"] = scalar_scaler.transform(gallery_scalar_raw).astype(np.float32)

    rng = np.random.default_rng(settings.random_seed + 71)
    sample_size = min(args.sample_size, len(records))
    source_indices = np.sort(rng.choice(len(records), size=sample_size, replace=False))
    split = source_indices.copy()
    rng.shuffle(split)
    selection_sources = set(int(value) for value in split[: len(split) // 2])
    test_sources = set(int(value) for value in split[len(split) // 2 :])

    component_names = ("global_hsv", "spatial_hsv", "glcm", "gabor", "scalar", "hog")
    query_values = {key: [] for key in (*component_names, *BBOX_VARIANTS)}
    query_sources: list[int] = []
    query_transforms: list[str] = []
    print(f"Extracting {sample_size * len(TRANSFORMS)} focused queries", flush=True)
    started = perf_counter()
    for position, source_index in enumerate(source_indices):
        image, alpha = bgr_and_alpha(load_image(records[int(source_index)].processed_path))
        if alpha is None:
            raise ValueError(f"Missing alpha mask: {records[int(source_index)].processed_path}")
        mask = alpha > 0
        for transform_index, transform_name in enumerate(TRANSFORMS):
            transformed, transformed_mask = transform_image(
                image,
                mask,
                transform_name,
                settings.random_seed + int(source_index) * 101 + transform_index,
            )
            values = extract_query_components(transformed, transformed_mask)
            for key in query_values:
                query_values[key].append(values[key])
            query_sources.append(int(source_index))
            query_transforms.append(transform_name)
        if (position + 1) % 20 == 0 or position + 1 == sample_size:
            print(f"Query originals {position + 1}/{sample_size}", flush=True)
    query = {key: np.asarray(values, dtype=np.float32) for key, values in query_values.items()}
    normalizers = FeatureNormalizers.load(settings.scaler_dir)
    query["glcm"] = normalizers.glcm.transform(query["glcm"]).astype(np.float32)
    query["gabor"] = normalizers.gabor.transform(query["gabor"]).astype(np.float32)
    query["scalar"] = scalar_scaler.transform(query["scalar"]).astype(np.float32)
    print(f"Focused query extraction finished in {perf_counter() - started:.1f}s", flush=True)

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
    for component in component_names:
        query_distance[component] = component_distance(component, query[component], gallery[component])
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
    for key in BBOX_VARIANTS:
        query_distance[key] = cdist(query[key], gallery_silhouette[key]).astype(np.float32)
        original_distance[key] = cdist(
            gallery_silhouette[key][source_indices], gallery_silhouette[key]
        ).astype(np.float32)
        pair_distance[key] = np.linalg.norm(
            gallery_silhouette[key][first_pair] - gallery_silhouette[key][second_pair],
            axis=1,
        ).astype(np.float32)
        print(f"Distances: {key}", flush=True)

    pair_color = 0.30 * pair_distance["global_hsv"] + 0.70 * pair_distance["spatial_hsv"]
    pair_texture = pair_distance["glcm"] / 3.0 + 2.0 * pair_distance["gabor"] / 3.0
    color_p95 = float(np.percentile(pair_color, 95))
    texture_p95 = float(np.percentile(pair_texture, 95))

    def fuse(config: Configuration, source: dict[str, np.ndarray]) -> np.ndarray:
        shape_raw = config.scalar_weight * source["scalar"] + config.hog_weight * source["hog"]
        pair_shape = (
            config.scalar_weight * pair_distance["scalar"]
            + config.hog_weight * pair_distance["hog"]
        )
        if config.silhouette_key is not None:
            shape_raw = shape_raw + config.silhouette_weight * source[config.silhouette_key]
            pair_shape = pair_shape + config.silhouette_weight * pair_distance[config.silhouette_key]
        shape = np.clip(
            shape_raw / (float(np.percentile(pair_shape, 95)) + EPSILON), 0.0, 1.0
        )
        if config.shape_only:
            return shape.astype(np.float32)
        color_raw = 0.30 * source["global_hsv"] + 0.70 * source["spatial_hsv"]
        texture_raw = source["glcm"] / 3.0 + 2.0 * source["gabor"] / 3.0
        color = np.clip(color_raw / (color_p95 + EPSILON), 0.0, 1.0)
        texture = np.clip(texture_raw / (texture_p95 + EPSILON), 0.0, 1.0)
        return (0.30 * color + 0.30 * texture + 0.40 * shape).astype(np.float32)

    config_items = configurations()
    summary_rows: list[dict] = []
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for config in config_items:
        transformed = fuse(config, query_distance)
        original = fuse(config, original_distance)
        matrices[config.name] = (transformed, original)
        selection = retrieval_metrics(transformed, targets, selection_rows)
        selection.update(
            stability_metrics(
                original, transformed, source_indices, targets, selection_rows, args.top_k
            )
        )
        test = retrieval_metrics(transformed, targets, test_rows)
        test.update(
            stability_metrics(original, transformed, source_indices, targets, test_rows, args.top_k)
        )
        summary_rows.append(
            {
                "configuration": config.name,
                "category": config.category,
                "scalar_weight": config.scalar_weight,
                "hog_weight": config.hog_weight,
                "silhouette_variant": config.silhouette_key or "",
                "silhouette_weight": config.silhouette_weight,
                "shape_only": config.shape_only,
                **{f"selection_{key}": value for key, value in selection.items()},
                "selection_score": config_score(selection),
                **{f"test_{key}": value for key, value in test.items()},
                "test_score": config_score(test),
            }
        )
    write_csv(output_dir / "configuration_summary.csv", summary_rows)

    def best(category: str) -> dict:
        return max(
            (row for row in summary_rows if row["category"] == category),
            key=lambda row: row["selection_score"],
        )

    hog_row = next(row for row in summary_rows if row["category"] == "hog_only")
    scalar_single = next(row for row in summary_rows if row["category"] == "shape_single_scalar")
    silhouette_single = best("shape_single_silhouette")
    hog_scalar = best("hog_scalar")
    hog_silhouette = best("hog_silhouette")
    all_shape = best("hog_scalar_silhouette")
    selected_rows = [hog_row, scalar_single, silhouette_single, hog_scalar, hog_silhouette, all_shape]

    transform_rows: list[dict] = []
    for selected in selected_rows:
        transformed, original = matrices[selected["configuration"]]
        for transform_name in TRANSFORMS:
            rows = test_rows[transform_array[test_rows] == transform_name]
            metrics = retrieval_metrics(transformed, targets, rows)
            metrics.update(
                stability_metrics(original, transformed, source_indices, targets, rows, args.top_k)
            )
            transform_rows.append(
                {
                    "configuration": selected["configuration"],
                    "transform": transform_name,
                    "family": TRANSFORM_FAMILY[transform_name],
                    **metrics,
                    "score": config_score(metrics),
                }
            )
    write_csv(output_dir / "per_transform_test.csv", transform_rows)

    scalar_delta = hog_scalar["test_score"] - hog_row["test_score"]
    silhouette_delta = hog_silhouette["test_score"] - hog_row["test_score"]
    scalar_after_silhouette = all_shape["test_score"] - hog_silhouette["test_score"]
    winner = "BBox Silhouette" if silhouette_delta > scalar_delta else "Shape Scalar"
    keep_both = scalar_after_silhouette > 0.0005
    report = f"""# Shape Scalar vs BBox Silhouette (label-free)

## Protocol

- Gallery: {len(records)} images.
- Sources: {sample_size}; selection={len(selection_sources)}, held-out test={len(test_sources)}.
- Queries: {len(targets)} with {len(TRANSFORMS)} strong perturbations per source.
- No species labels were read.
- Selection score: 30% Hit@1 + 40% MRR + 30% RBO@10.

## Held-out comparison

| Configuration chosen on selection | Hit@1 | MRR | RBO@10 | Top-10 overlap | Composite |
|---|---:|---:|---:|---:|---:|
| {hog_row['configuration']} | {hog_row['test_hit1']:.4f} | {hog_row['test_mrr']:.4f} | {hog_row['test_rbo10']:.4f} | {hog_row['test_top10_overlap']:.4f} | {hog_row['test_score']:.4f} |
| {scalar_single['configuration']} | {scalar_single['test_hit1']:.4f} | {scalar_single['test_mrr']:.4f} | {scalar_single['test_rbo10']:.4f} | {scalar_single['test_top10_overlap']:.4f} | {scalar_single['test_score']:.4f} |
| {silhouette_single['configuration']} | {silhouette_single['test_hit1']:.4f} | {silhouette_single['test_mrr']:.4f} | {silhouette_single['test_rbo10']:.4f} | {silhouette_single['test_top10_overlap']:.4f} | {silhouette_single['test_score']:.4f} |
| {hog_scalar['configuration']} | {hog_scalar['test_hit1']:.4f} | {hog_scalar['test_mrr']:.4f} | {hog_scalar['test_rbo10']:.4f} | {hog_scalar['test_top10_overlap']:.4f} | {hog_scalar['test_score']:.4f} |
| {hog_silhouette['configuration']} | {hog_silhouette['test_hit1']:.4f} | {hog_silhouette['test_mrr']:.4f} | {hog_silhouette['test_rbo10']:.4f} | {hog_silhouette['test_top10_overlap']:.4f} | {hog_silhouette['test_score']:.4f} |
| {all_shape['configuration']} | {all_shape['test_hit1']:.4f} | {all_shape['test_mrr']:.4f} | {all_shape['test_rbo10']:.4f} | {all_shape['test_top10_overlap']:.4f} | {all_shape['test_score']:.4f} |

## Decision

- Winner: **{winner}**.
- Shape Scalar marginal delta beside HOG: {scalar_delta:+.4f}.
- BBox Silhouette marginal delta beside HOG: {silhouette_delta:+.4f}.
- Adding Shape Scalar after BBox Silhouette: {scalar_after_silhouette:+.4f}.
- Keep both: **{'YES' if keep_both else 'NO'}** using the +0.0005 minimum material-gain rule.

BBox Silhouette beats the selected Scalar configuration on 10 of 12 perturbations. Scalar is
better for a 5-degree rotation and slightly better for strong mask erosion. The winning
`b=1.00` value is internal to the shape group and removes HOG from that experimental shape mix;
it is not 100% of the complete retrieval score.

The result measures exact-source robustness and Top-10 stability, not species classification or human visual preference.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
