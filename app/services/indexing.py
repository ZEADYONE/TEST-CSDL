"""Recursive, fault-tolerant dataset indexing into PostgreSQL."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from app.config import Settings, get_settings
from app.models import ImageFeature, ImageRecord
from app.services.feature_extractor import FeatureBundle
from app.services.pipeline import BirdCBIRPipeline
from app.services.preprocessing import (
    YoloBirdSegmenter,
    load_image,
    rgba_image,
    write_image,
)


LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class PendingImage:
    source: Path
    root: Path
    species: str | None
    width: int
    height: int
    processed_path: Path
    features: FeatureBundle


@dataclass(frozen=True)
class IndexSummary:
    discovered: int
    indexed: int
    failed: int


def collect_images(roots: list[Path]) -> list[tuple[Path, Path]]:
    discovered: dict[Path, Path] = {}
    for root in roots:
        resolved = root.resolve()
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        for path in resolved.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                discovered.setdefault(path.resolve(), resolved)
    return sorted(discovered.items(), key=lambda item: str(item[0]).casefold())


def _species(path: Path, root: Path) -> str | None:
    relative = path.relative_to(root)
    return path.parent.name if len(relative.parts) > 1 else None


def _processed_path(source: Path, settings: Settings) -> Path:
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:16]
    return settings.static_dir / "dataset" / f"{digest}_{source.stem}.png"


def index_dataset(
    session: Session,
    roots: list[Path],
    *,
    already_preprocessed: bool,
    settings: Settings | None = None,
) -> IndexSummary:
    """Extract all inputs, fit dataset-only normalization, and upsert records."""

    config = settings or get_settings()
    config.ensure_directories()
    files = collect_images(roots)
    segmenter = None if already_preprocessed else YoloBirdSegmenter(config)
    pipeline = BirdCBIRPipeline(config, segmenter=segmenter)
    pending: list[PendingImage] = []
    failures = 0

    for source, root in tqdm(files, desc="Extracting features", unit="image"):
        try:
            raw = load_image(source)
            original_height, original_width = raw.shape[:2]
            extracted = pipeline.process(
                raw,
                already_preprocessed=already_preprocessed,
            )
            processed = extracted.processed
            output = _processed_path(source, config)
            write_image(output, rgba_image(processed.image, processed.mask))
            pending.append(
                PendingImage(
                    source.resolve(),
                    root,
                    _species(source, root),
                    original_width,
                    original_height,
                    output.resolve(),
                    extracted.features,
                )
            )
        except Exception:
            failures += 1
            LOGGER.exception("Could not index %s", source)

    if len(pending) < 2:
        raise RuntimeError(
            "Fewer than two valid images remain; indexing cannot fit distance statistics"
        )
    normalized = pipeline.fit_dataset([item.features for item in pending])

    existing = {
        record.file_path: record
        for record in session.scalars(
            select(ImageRecord).where(
                ImageRecord.file_path.in_([str(item.source) for item in pending])
            )
        )
    }
    for item, bundle in tqdm(
        list(zip(pending, normalized)), desc="Writing PostgreSQL", unit="image"
    ):
        key = str(item.source)
        record = existing.get(key)
        if record is None:
            record = ImageRecord(file_path=key)
            session.add(record)
        record.file_name = item.source.name
        record.processed_path = str(item.processed_path)
        record.species = item.species
        record.width = item.width
        record.height = item.height
        if record.feature is None:
            record.feature = ImageFeature(feature_version=config.feature_version)
        record.feature.color_vector = bundle.color.astype(float).tolist()
        record.feature.texture_vector = bundle.texture.astype(float).tolist()
        record.feature.shape_vector = bundle.shape.astype(float).tolist()
        record.feature.feature_version = config.feature_version
    session.commit()
    return IndexSummary(len(files), len(pending), failures)
