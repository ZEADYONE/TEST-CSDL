from __future__ import annotations

import cv2
import numpy as np

from app.config import Settings
from app.services.pipeline import BirdCBIRPipeline


def _with_alpha(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask.astype(np.uint8) * 255
    return bgra


def test_pipeline_runs_preprocess_extract_fit_and_rank(bird_sample, tmp_path):
    image, mask = bird_sample
    settings = Settings(
        target_size=384,
        scaler_dir=tmp_path / "models",
        distance_stats_path=tmp_path / "models" / "distance_stats.json",
        static_dir=tmp_path / "static",
        debug_dir=tmp_path / "debug",
        distance_sample_pairs=10,
    )
    pipeline = BirdCBIRPipeline(settings)

    first = pipeline.process(_with_alpha(image, mask), already_preprocessed=True)
    changed = image.copy()
    changed[mask, 0] = 255 - changed[mask, 0]
    changed[mask, 1] = changed[mask, 1] // 3
    second = pipeline.process(_with_alpha(changed, mask), already_preprocessed=True)

    gallery = pipeline.fit_dataset([first.features, second.features])
    query = pipeline.normalize(first.features)
    matches = pipeline.rank(
        query,
        gallery,
        mask_quality=first.processed.quality.quality_score,
        top_k=2,
    )

    assert matches[0].candidate_index == 0
    assert matches[0].similarity == 1.0
    assert len(matches) == 2
    assert (settings.scaler_dir / "glcm_scaler.joblib").is_file()
    assert (settings.scaler_dir / "gabor_scaler.joblib").is_file()
    assert settings.distance_stats_path.is_file()


def test_pipeline_rejects_non_positive_top_k(bird_sample, tmp_path):
    image, mask = bird_sample
    settings = Settings(
        scaler_dir=tmp_path / "models",
        distance_stats_path=tmp_path / "models" / "distance_stats.json",
        static_dir=tmp_path / "static",
        debug_dir=tmp_path / "debug",
        distance_sample_pairs=1,
    )
    pipeline = BirdCBIRPipeline(settings)
    extracted = pipeline.process(_with_alpha(image, mask), already_preprocessed=True)
    gallery = pipeline.fit_dataset([extracted.features, extracted.features], persist=False)

    try:
        pipeline.rank(gallery[0], gallery, mask_quality=1.0, top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for top_k=0")
