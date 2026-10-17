"""Run one image through the complete production CBIR pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import joinedload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import ImageFeature, ImageRecord  # noqa: E402
from app.services.feature_extractor import FeatureBundle  # noqa: E402
from app.services.pipeline import BirdCBIRPipeline  # noqa: E402
from app.services.preprocessing import load_image  # noqa: E402
from app.services.similarity import effective_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Query image path")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--already-preprocessed",
        action="store_true",
        help="Read the alpha/constant-background mask instead of running YOLO-Seg",
    )
    parser.add_argument("--flip-horizontal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    settings = get_settings()
    pipeline = BirdCBIRPipeline.from_saved_artifacts(settings)
    extracted = pipeline.process(
        load_image(args.image),
        already_preprocessed=args.already_preprocessed,
        flip_horizontal=args.flip_horizontal,
    )
    query = pipeline.normalize(extracted.features)

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
        if not records:
            raise LookupError("The database has no current-version indexed images")
        candidates = [
            FeatureBundle(
                np.asarray(record.feature.color_vector, dtype=np.float32),
                np.asarray(record.feature.texture_vector, dtype=np.float32),
                np.asarray(record.feature.shape_vector, dtype=np.float32),
            ).validate()
            for record in records
        ]
        matches = pipeline.rank(
            query,
            candidates,
            mask_quality=extracted.processed.quality.quality_score,
            top_k=args.top_k,
        )

    result = {
        "pipeline": [
            "load",
            "segment_or_alpha_mask",
            "clean_crop_resize_pad_384",
            "extract_color_texture_shape",
            "normalize_glcm_gabor",
            "per_group_distance_scaling",
            "weighted_visual_ranking",
        ],
        "feature_version": settings.feature_version,
        "query": str(args.image.resolve()),
        "mask_quality": extracted.processed.quality.quality_score,
        "effective_weights": effective_weights(
            extracted.processed.quality.quality_score, settings
        ),
        "dimensions": {
            "color": int(query.color.size),
            "texture": int(query.texture.size),
            "shape": int(query.shape.size),
            "total": int(query.color.size + query.texture.size + query.shape.size),
        },
        "results": [
            {
                "rank": rank,
                "image_id": records[match.candidate_index].id,
                "file_name": records[match.candidate_index].file_name,
                "species_metadata": records[match.candidate_index].species,
                "similarity": match.similarity,
                "color_similarity": match.color_similarity,
                "texture_similarity": match.texture_similarity,
                "shape_similarity": match.shape_similarity,
            }
            for rank, match in enumerate(matches, start=1)
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
