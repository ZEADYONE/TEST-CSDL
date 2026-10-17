"""Single orchestration pipeline for indexing and content-based retrieval.

The descriptor implementations remain small, independently testable stages.
This module owns their execution order so dataset indexing and online queries
cannot silently drift into different preprocessing, normalization, or ranking
paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.config import Settings, get_settings
from app.services.feature_extractor import BirdFeatureExtractor, FeatureBundle
from app.services.normalization import FeatureNormalizers
from app.services.preprocessing import ProcessedBird, YoloBirdSegmenter, preprocess_array
from app.services.similarity import (
    DistanceScaling,
    component_distances,
    effective_weights,
    fit_distance_scaling,
    similarities,
)


@dataclass(frozen=True)
class ExtractedBird:
    """Output of the preprocessing and raw-feature stages."""

    processed: ProcessedBird
    features: FeatureBundle


@dataclass(frozen=True)
class RankedFeature:
    """Score breakdown for one candidate, addressed by its input position."""

    candidate_index: int
    similarity: float
    color_similarity: float
    texture_similarity: float
    shape_similarity: float


class BirdCBIRPipeline:
    """Run the complete handcrafted CBIR pipeline in one consistent order.

    Dataset flow::

        image -> preprocess -> extract -> fit/transform -> distance statistics

    Query flow::

        image -> preprocess -> extract -> transform -> compare -> rank Top-K
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        segmenter: YoloBirdSegmenter | None = None,
        normalizers: FeatureNormalizers | None = None,
        distance_scaling: DistanceScaling | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.segmenter = segmenter
        self.extractor = BirdFeatureExtractor()
        self.normalizers = normalizers
        self.distance_scaling = distance_scaling

    @classmethod
    def from_saved_artifacts(
        cls,
        settings: Settings | None = None,
        *,
        segmenter: YoloBirdSegmenter | None = None,
    ) -> "BirdCBIRPipeline":
        """Build a query-ready pipeline from dataset-fitted artifacts."""

        config = settings or get_settings()
        return cls(
            config,
            segmenter=segmenter,
            normalizers=FeatureNormalizers.load(config.scaler_dir),
            distance_scaling=DistanceScaling.load(config.distance_stats_path),
        )

    def process(
        self,
        image: np.ndarray,
        *,
        already_preprocessed: bool,
        flip_horizontal: bool = False,
    ) -> ExtractedBird:
        """Preprocess one image and extract all raw feature groups."""

        processed = preprocess_array(
            image,
            already_preprocessed=already_preprocessed,
            flip_horizontal=flip_horizontal,
            segmenter=self.segmenter,
            settings=self.settings,
        )
        return ExtractedBird(
            processed=processed,
            features=self.extractor.extract(processed.image, processed.mask),
        )

    def fit_dataset(
        self,
        bundles: Sequence[FeatureBundle],
        *,
        persist: bool = True,
    ) -> list[FeatureBundle]:
        """Fit dataset-only artifacts and return normalized gallery vectors."""

        values = list(bundles)
        if len(values) < 2:
            raise ValueError("At least two feature bundles are required to fit the pipeline")
        self.normalizers = FeatureNormalizers.fit(values)
        normalized = [self.normalizers.transform(bundle) for bundle in values]
        self.distance_scaling = fit_distance_scaling(normalized, settings=self.settings)
        if persist:
            self.settings.ensure_directories()
            self.normalizers.save(self.settings.scaler_dir)
            self.distance_scaling.save(self.settings.distance_stats_path)
        return normalized

    def normalize(self, bundle: FeatureBundle) -> FeatureBundle:
        """Apply the fitted numerical scalers to one feature bundle."""

        if self.normalizers is None:
            raise RuntimeError("Pipeline normalizers are not fitted or loaded")
        return self.normalizers.transform(bundle)

    def rank(
        self,
        query: FeatureBundle,
        candidates: Sequence[FeatureBundle],
        *,
        mask_quality: float,
        top_k: int,
    ) -> list[RankedFeature]:
        """Compare a normalized query with normalized candidates and rank them."""

        if self.distance_scaling is None:
            raise RuntimeError("Pipeline distance scaling is not fitted or loaded")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        weights = effective_weights(mask_quality, self.settings)
        ranked: list[RankedFeature] = []
        for index, candidate in enumerate(candidates):
            score = similarities(
                component_distances(query, candidate, self.settings),
                self.distance_scaling,
                weights,
            )
            ranked.append(
                RankedFeature(
                    candidate_index=index,
                    similarity=score[0],
                    color_similarity=score[1],
                    texture_similarity=score[2],
                    shape_similarity=score[3],
                )
            )
        ranked.sort(key=lambda item: item.similarity, reverse=True)
        return ranked[: min(top_k, len(ranked))]
