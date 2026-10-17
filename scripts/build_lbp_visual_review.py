"""Build label-free contact sheets for visual review of current LBP ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np
from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import SessionLocal  # noqa: E402
from app.models import ImageRecord  # noqa: E402
from app.services.preprocessing import bgr_and_alpha, load_image, write_image  # noqa: E402
from scripts.evaluate_visual_feature_selection import transform_image  # noqa: E402


CELL_WIDTH = 300
CELL_HEIGHT = 300
IMAGE_SIZE = 250
COLS = 4


def fit_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / max(width, 1), size / max(height, 1))
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized[:, :, :3]
    return canvas


def card(image: np.ndarray, title: str, *, query: bool = False) -> np.ndarray:
    canvas = np.full((CELL_HEIGHT, CELL_WIDTH, 3), 255, dtype=np.uint8)
    fitted = fit_image(image)
    x = (CELL_WIDTH - IMAGE_SIZE) // 2
    canvas[38 : 38 + IMAGE_SIZE, x : x + IMAGE_SIZE] = fitted
    color = (30, 130, 40) if query else (35, 35, 35)
    thickness = 4 if query else 1
    cv2.rectangle(canvas, (x, 38), (x + IMAGE_SIZE, 38 + IMAGE_SIZE), color, thickness)
    cv2.putText(
        canvas,
        title,
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2 if query else 1,
        cv2.LINE_AA,
    )
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratings",
        type=Path,
        default=PROJECT_ROOT / "results" / "proposed_feature_experiments" / "manual_visual_ratings.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "lbp_visual_review",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with args.ratings.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    relevant = [
        row
        for row in rows
        if row["configuration"] in {"Current Full", "Current Full - LBP"}
    ]
    query_keys = sorted({(row["query_source_id"], row["transformation"]) for row in relevant})
    ids = {int(row["query_source_id"]) for row in relevant}
    ids.update(int(row["result_image_id"]) for row in relevant)
    with SessionLocal() as session:
        records = {
            record.id: record
            for record in session.scalars(select(ImageRecord).where(ImageRecord.id.in_(ids))).all()
        }

    comparison_rows: list[dict] = []
    for query_number, (source_id_text, transformation) in enumerate(query_keys, start=1):
        source_id = int(source_id_text)
        full = sorted(
            (
                row
                for row in relevant
                if row["query_source_id"] == source_id_text
                and row["transformation"] == transformation
                and row["configuration"] == "Current Full"
            ),
            key=lambda row: int(row["result_rank"]),
        )
        no_lbp = sorted(
            (
                row
                for row in relevant
                if row["query_source_id"] == source_id_text
                and row["transformation"] == transformation
                and row["configuration"] == "Current Full - LBP"
            ),
            key=lambda row: int(row["result_rank"]),
        )
        full_ids = [int(row["result_image_id"]) for row in full]
        no_lbp_ids = [int(row["result_image_id"]) for row in no_lbp]
        same_order = full_ids == no_lbp_ids
        comparison_rows.append(
            {
                "query_number": query_number,
                "query_source_id": source_id,
                "transformation": transformation,
                "same_top10_order": same_order,
                "top10_overlap_count": len(set(full_ids) & set(no_lbp_ids)),
            }
        )

        source = records[source_id]
        raw = load_image(source.processed_path)
        image, alpha = bgr_and_alpha(raw)
        if alpha is None:
            raise ValueError(f"Processed image lacks alpha: {source.processed_path}")
        query_image, _ = transform_image(image, alpha > 0, transformation)
        cards = [card(query_image, f"QUERY {query_number}: {transformation}", query=True)]
        for rank, image_id in enumerate(full_ids, start=1):
            candidate_raw = load_image(records[image_id].processed_path)
            candidate, _ = bgr_and_alpha(candidate_raw)
            cards.append(card(candidate, f"#{rank}  image_id={image_id}"))

        while len(cards) % COLS:
            cards.append(np.full((CELL_HEIGHT, CELL_WIDTH, 3), 245, dtype=np.uint8))
        sheet_rows = [
            np.hstack(cards[start : start + COLS])
            for start in range(0, len(cards), COLS)
        ]
        sheet = np.vstack(sheet_rows)
        write_image(output_dir / f"query_{query_number:02d}_{transformation}.png", sheet)

    with (output_dir / "top10_order_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Saved {len(query_keys)} contact sheets to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
