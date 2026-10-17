"""Find and visualize queries whose Top-10 changes most when current LBP is removed."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import joinedload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import ImageFeature, ImageRecord  # noqa: E402
from app.services.normalization import FeatureNormalizers  # noqa: E402
from app.services.preprocessing import bgr_and_alpha, load_image, write_image  # noqa: E402
from scripts.build_lbp_visual_review import card  # noqa: E402
from scripts.evaluate_proposed_features import rbo_at_k  # noqa: E402
from scripts.evaluate_visual_feature_selection import (  # noqa: E402
    COMPONENTS,
    TRANSFORMS,
    atomic_from_stored,
    component_distance,
    extract_atomic,
    transform_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "lbp_visual_review" / "sensitive_cases",
    )
    return parser.parse_args()


def load_p95() -> tuple[float, float, float, float]:
    path = PROJECT_ROOT / "results" / "proposed_feature_experiments" / "configuration_summary.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["configuration"]: row for row in csv.DictReader(handle)}
    baseline = rows["Current Full"]
    no_lbp = rows["Current Full - LBP"]
    return (
        float(baseline["p95_color"]),
        float(baseline["p95_texture"]),
        float(no_lbp["p95_texture"]),
        float(baseline["p95_shape"]),
    )


def rank_query(
    query: dict[str, np.ndarray],
    gallery: dict[str, np.ndarray],
    source_index: int,
    p95: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    color_p95, texture_p95, no_lbp_texture_p95, shape_p95 = p95
    distance = {
        component: component_distance(
            component,
            query[component].reshape(1, -1),
            gallery[component],
        )[0]
        for component in ("global_hsv", "spatial_hsv", "lbp", "glcm", "gabor", "scalar", "hog")
    }
    color = np.clip(
        (0.30 * distance["global_hsv"] + 0.70 * distance["spatial_hsv"]) / color_p95,
        0.0,
        1.0,
    )
    texture = np.clip(
        (0.10 * distance["lbp"] + 0.30 * distance["glcm"] + 0.60 * distance["gabor"])
        / texture_p95,
        0.0,
        1.0,
    )
    texture_no_lbp = np.clip(
        ((0.30 / 0.90) * distance["glcm"] + (0.60 / 0.90) * distance["gabor"])
        / no_lbp_texture_p95,
        0.0,
        1.0,
    )
    shape = np.clip(
        (0.20 * distance["scalar"] + 0.80 * distance["hog"]) / shape_p95,
        0.0,
        1.0,
    )
    full = 0.30 * color + 0.30 * texture + 0.40 * shape
    no_lbp = 0.30 * color + 0.30 * texture_no_lbp + 0.40 * shape
    full_order = np.argsort(full, kind="stable")
    no_lbp_order = np.argsort(no_lbp, kind="stable")
    return (
        full_order[full_order != source_index][:10],
        no_lbp_order[no_lbp_order != source_index][:10],
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    settings = get_settings()
    output_dir = args.output_dir.resolve()
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
    gallery = atomic_from_stored(records)
    normalizers = FeatureNormalizers.load(settings.scaler_dir)
    p95 = load_p95()
    rng = np.random.default_rng(settings.random_seed)
    source_indices = np.sort(
        rng.choice(len(records), size=min(args.sample_size, len(records)), replace=False)
    )
    comparisons: list[dict] = []
    total = len(source_indices) * len(TRANSFORMS)
    completed = 0
    for source_index in source_indices:
        raw = load_image(records[int(source_index)].processed_path)
        image, alpha = bgr_and_alpha(raw)
        if alpha is None:
            raise ValueError(f"Processed image lacks alpha: {records[int(source_index)].processed_path}")
        mask = alpha > 0
        for transformation in TRANSFORMS:
            transformed, transformed_mask = transform_image(image, mask, transformation)
            atomic, _ = extract_atomic(transformed, transformed_mask)
            atomic["glcm"] = normalizers.glcm.transform(atomic["glcm"].reshape(1, -1))[0]
            atomic["gabor"] = normalizers.gabor.transform(atomic["gabor"].reshape(1, -1))[0]
            atomic["scalar"] = normalizers.shape_scalars.transform(atomic["scalar"].reshape(1, -1))[0]
            full, no_lbp = rank_query(atomic, gallery, int(source_index), p95)
            overlap = len(set(full.tolist()) & set(no_lbp.tolist()))
            comparisons.append(
                {
                    "source_index": int(source_index),
                    "query_source_id": records[int(source_index)].id,
                    "transformation": transformation,
                    "overlap_count": overlap,
                    "config_rbo10": rbo_at_k(full, no_lbp),
                    "changed_positions": int(np.sum(full != no_lbp)),
                    "full_indices": full,
                    "no_lbp_indices": no_lbp,
                }
            )
            completed += 1
        if completed % 140 == 0 or completed == total:
            print(f"Rank comparisons: {completed}/{total}", flush=True)

    ordered = sorted(
        comparisons,
        key=lambda row: (row["overlap_count"], row["config_rbo10"], -row["changed_positions"]),
    )
    selected: list[dict] = []
    used_sources: set[int] = set()
    for row in ordered:
        source_index = int(row["source_index"])
        if source_index in used_sources:
            continue
        selected.append(row)
        used_sources.add(source_index)
        if len(selected) == args.cases:
            break

    summary_rows: list[dict] = []
    rating_template: list[dict] = []
    for case_number, row in enumerate(selected, start=1):
        source_index = int(row["source_index"])
        transformation = str(row["transformation"])
        raw = load_image(records[source_index].processed_path)
        image, alpha = bgr_and_alpha(raw)
        assert alpha is not None
        query_image, _ = transform_image(image, alpha > 0, transformation)
        full = np.asarray(row["full_indices"], dtype=int)
        no_lbp = np.asarray(row["no_lbp_indices"], dtype=int)
        full_set = set(full.tolist())
        no_lbp_set = set(no_lbp.tolist())
        cards = [card(query_image, f"QUERY {case_number}: {transformation}", query=True)]
        cards.extend([np.full((300, 300, 3), 245, dtype=np.uint8) for _ in range(4)])
        for config_name, ranking, other_set in (
            ("FULL", full, no_lbp_set),
            ("NO LBP", no_lbp, full_set),
        ):
            for rank, candidate in enumerate(ranking, start=1):
                candidate_raw = load_image(records[int(candidate)].processed_path)
                candidate_image, _ = bgr_and_alpha(candidate_raw)
                marker = " NEW" if int(candidate) not in other_set else ""
                cards.append(card(candidate_image, f"{config_name} #{rank}{marker} id={records[int(candidate)].id}"))
                rating_template.append(
                    {
                        "case_number": case_number,
                        "query_source_id": records[source_index].id,
                        "transformation": transformation,
                        "configuration": config_name,
                        "result_rank": rank,
                        "result_image_id": records[int(candidate)].id,
                        "is_unique_to_configuration": int(candidate) not in other_set,
                        "texture_rating_1_to_5": "",
                        "overall_rating_1_to_5": "",
                        "notes": "",
                    }
                )
        while len(cards) % 5:
            cards.append(np.full((300, 300, 3), 245, dtype=np.uint8))
        sheet = np.vstack(
            [np.hstack(cards[start : start + 5]) for start in range(0, len(cards), 5)]
        )
        write_image(output_dir / f"sensitive_{case_number:02d}_{transformation}.png", sheet)
        summary_rows.append(
            {
                "case_number": case_number,
                "query_source_id": records[source_index].id,
                "transformation": transformation,
                "top10_overlap_count": row["overlap_count"],
                "config_rbo10": row["config_rbo10"],
                "changed_positions": row["changed_positions"],
            }
        )

    write_csv(output_dir / "sensitive_query_summary.csv", summary_rows)
    write_csv(output_dir / "sensitive_visual_rating_template.csv", rating_template)
    compact_rows = [
        {
            "query_source_id": row["query_source_id"],
            "transformation": row["transformation"],
            "top10_overlap_count": row["overlap_count"],
            "config_rbo10": row["config_rbo10"],
            "changed_positions": row["changed_positions"],
        }
        for row in comparisons
    ]
    write_csv(output_dir / "all_query_rank_changes.csv", compact_rows)
    print(f"Saved {len(selected)} most LBP-sensitive cases to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
