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
from app.services.feature_extractor import FeatureBundle
from app.services.debug_visualization import save_debug_outputs
from app.services.pipeline import BirdCBIRPipeline
from app.services.preprocessing import (
    ProcessedBird,
    decode_image,
    rgba_image,
    write_image,
)
from app.services.similarity import effective_weights


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
    pipeline = BirdCBIRPipeline.from_saved_artifacts(config)
    extracted = pipeline.process(
        raw,
        already_preprocessed=False,
        flip_horizontal=flip_horizontal,
    )
    processed = extracted.processed
    query = pipeline.normalize(extracted.features)

    records = session.scalars(
        select(ImageRecord)
        .options(joinedload(ImageRecord.feature))
        .join(ImageFeature)
        .where(ImageFeature.feature_version == config.feature_version)
    ).all()
    if not records:
        raise LookupError("The database has no indexed images")
    candidates: list[FeatureBundle] = []
    for record in records:
        feature = record.feature
        candidates.append(
            FeatureBundle(
                np.asarray(feature.color_vector, dtype=np.float32),
                np.asarray(feature.texture_vector, dtype=np.float32),
                np.asarray(feature.shape_vector, dtype=np.float32),
            ).validate()
        )
    matches = pipeline.rank(
        query,
        candidates,
        mask_quality=processed.quality.quality_score,
        top_k=top_k,
    )
    ranked: list[SearchResult] = []
    for match in matches:
        record = records[match.candidate_index]
        ranked.append(
            SearchResult(
                image_id=record.id,
                file_name=record.file_name,
                species=record.species,
                image_url=f"/api/images/{record.id}",
                similarity=match.similarity,
                color_similarity=match.color_similarity,
                texture_similarity=match.texture_similarity,
                shape_similarity=match.shape_similarity,
            )
        )
    urls = _save_query_artifacts(raw, processed, config)
    debug = (
        SearchDebug(
            mask_quality=processed.quality.quality_score,
            effective_weights=effective_weights(processed.quality.quality_score, config),
        )
        if config.debug_features
        else None
    )
    return SearchResponse(
        query_original_url=urls[0],
        query_processed_url=urls[1],
        query_mask_url=urls[2],
        mask_quality=processed.quality.quality_score,
        results=ranked,
        debug=debug,
    )
