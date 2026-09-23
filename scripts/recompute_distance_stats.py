"""Recompute distance percentiles from vectors already stored in PostgreSQL."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import joinedload

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import ImageFeature, ImageRecord  # noqa: E402
from app.services.feature_extractor import FeatureBundle  # noqa: E402
from app.services.similarity import fit_distance_scaling  # noqa: E402


def main() -> int:
    settings = get_settings()
    with SessionLocal() as session:
        records = list(
            session.scalars(
                select(ImageRecord)
                .options(joinedload(ImageRecord.feature))
                .join(ImageFeature)
                .where(ImageFeature.feature_version == settings.feature_version)
            ).all()
        )
    bundles = [
        FeatureBundle(
            np.asarray(record.feature.color_vector, dtype=np.float32),
            np.asarray(record.feature.texture_vector, dtype=np.float32),
            np.asarray(record.feature.shape_vector, dtype=np.float32),
        ).validate()
        for record in records
    ]
    scaling = fit_distance_scaling(bundles, settings=settings)
    scaling.save(settings.distance_stats_path)
    print(json.dumps(asdict(scaling), indent=2))
    print(f"Saved: {settings.distance_stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
