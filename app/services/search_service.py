"""End-to-end query preprocessing and exhaustive CBIR search."""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.config import Settings, get_settings
from app.models import ImageFeature, ImageRecord
from app.schemas.search import SearchDebug, SearchResponse, SearchResult
from app.services.feature_extractor import BirdFeatureExtractor, FeatureBundle
from app.services.debug_visualization import save_debug_outputs
from app.services.normalization import FeatureNormalizers
from app.services.preprocessing import (
    ProcessedBird,
    decode_image,
    preprocess_array,
    rgba_image,
    write_image,
)
from app.services.similarity import (
    DistanceScaling,
    component_distances,
    effective_weights,
    similarities,
)


def _save_query_artifacts(
    original: np.ndarray,
    processed: ProcessedBird,
    settings: Settings,
) -> tuple[str, str, str]:
    token = uuid.uuid4().hex
    directory = settings.static_dir / "queries" / token
    original_path = directory / "original.png"
    processed_path = directory / "processed.png"
    mask_path = directory / "mask.png"
    original_bgr = original
    if original.ndim == 2:
        original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    elif original.shape[2] == 4:
        original_bgr = original[:, :, :3]
    write_image(original_path, original_bgr)
    write_image(processed_path, rgba_image(processed.image, processed.mask))
    write_image(mask_path, processed.mask.astype(np.uint8) * 255)
    if settings.debug_features:
        save_debug_outputs(processed, settings.debug_dir / token)
    prefix = f"/static/queries/{token}"
    return f"{prefix}/original.png", f"{prefix}/processed.png", f"{prefix}/mask.png"


def search_similar_images(
    session: Session,
    image_data: bytes,
    *,
    top_k: int = 10,
    flip_horizontal: bool = False,
    settings: Settings | None = None,
) -> SearchResponse:
    """Search every current-version database vector and return descending similarities."""

    config = settings or get_settings()
    config.ensure_directories()
    raw = decode_image(image_data)
    processed = preprocess_array(
        raw,
        already_preprocessed=False,
        flip_horizontal=flip_horizontal,
        settings=config,
    )
    raw_bundle = BirdFeatureExtractor().extract(processed.image, processed.mask)
    query = FeatureNormalizers.load(config.scaler_dir).transform(raw_bundle)
    scaling = DistanceScaling.load(config.distance_stats_path)
    weights = effective_weights(processed.quality.quality_score, config)

    records = session.scalars(
        select(ImageRecord)
        .options(joinedload(ImageRecord.feature))
        .join(ImageFeature)
        .where(ImageFeature.feature_version == config.feature_version)
    ).all()
    if not records:
        raise LookupError("The database has no indexed images")
    ranked: list[SearchResult] = []
    for record in records:
        feature = record.feature
        candidate = FeatureBundle(
            np.asarray(feature.color_vector, dtype=np.float32),
            np.asarray(feature.texture_vector, dtype=np.float32),
            np.asarray(feature.shape_vector, dtype=np.float32),
        ).validate()
        scores = similarities(component_distances(query, candidate, config), scaling, weights)
        ranked.append(
            SearchResult(
                image_id=record.id,
                file_name=record.file_name,
                species=record.species,
                image_url=f"/api/images/{record.id}",
                similarity=scores[0],
                color_similarity=scores[1],
                texture_similarity=scores[2],
                shape_similarity=scores[3],
            )
        )
    ranked.sort(key=lambda item: item.similarity, reverse=True)
    urls = _save_query_artifacts(raw, processed, config)
    debug = (
        SearchDebug(mask_quality=processed.quality.quality_score, effective_weights=weights)
        if config.debug_features
        else None
    )
    return SearchResponse(
        query_original_url=urls[0],
        query_processed_url=urls[1],
        query_mask_url=urls[2],
        mask_quality=processed.quality.quality_score,
        results=ranked[:top_k],
        debug=debug,
    )
