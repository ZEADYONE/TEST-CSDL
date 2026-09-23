"""Per-group distances, robust scaling, and quality-aware score fusion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.config import Settings, get_settings
from app.services.color_features import JOINT_DIM, SPATIAL_REGION_DIM, SPATIAL_REGIONS
from app.services.feature_extractor import FeatureBundle
from app.services.shape_features import SHAPE_DIM as GEOMETRY_DIM, SILHOUETTE_DIM
from app.services.texture_features import LBP_DIM, TEXTURE_DIM as BASE_TEXTURE_DIM


EPSILON = 1e-10


def chi_square_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    return float(0.5 * np.sum((first - second) ** 2 / (first + second + EPSILON)))


def euclidean_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)))


@dataclass(frozen=True)
class GroupDistances:
    color: float
    texture: float
    shape: float


@dataclass(frozen=True)
class DistanceRange:
    p5: float
    p95: float

    def normalize(self, value: float) -> float:
        """Scale distance against the robust upper percentile while preserving rank.

        Subtracting p5 made every distance in the closest five percent exactly
        zero after clipping. That is useful for outlier visualization but loses
        the fine ordering needed by Top-K retrieval. Anchoring true distance
        zero at normalized zero keeps identical images at similarity 1 and uses
        p95 to keep large outliers from controlling the scale. ``p5`` remains
        persisted as a useful dataset diagnostic.
        """

        return float(np.clip(value / (self.p95 + EPSILON), 0.0, 1.0))


@dataclass(frozen=True)
class DistanceScaling:
    color: DistanceRange
    texture: DistanceRange
    shape: DistanceRange

    def normalize(self, distances: GroupDistances) -> GroupDistances:
        return GroupDistances(
            self.color.normalize(distances.color),
            self.texture.normalize(distances.texture),
            self.shape.normalize(distances.shape),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DistanceScaling":
        if not path.is_file():
            raise FileNotFoundError("Distance statistics are missing. Run scripts/index_dataset.py first.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(*(DistanceRange(**payload[name]) for name in ("color", "texture", "shape")))


def component_distances(
    query: FeatureBundle,
    candidate: FeatureBundle,
    settings: Settings | None = None,
) -> GroupDistances:
    """Compute color, texture, and shape distances without cross-group concatenation."""

    config = settings or get_settings()
    joint_color = chi_square_distance(query.color[:JOINT_DIM], candidate.color[:JOINT_DIM])
    spatial_color = float(
        np.mean(
            [
                chi_square_distance(
                    query.color[
                        JOINT_DIM
                        + region * SPATIAL_REGION_DIM : JOINT_DIM
                        + (region + 1) * SPATIAL_REGION_DIM
                    ],
                    candidate.color[
                        JOINT_DIM
                        + region * SPATIAL_REGION_DIM : JOINT_DIM
                        + (region + 1) * SPATIAL_REGION_DIM
                    ],
                )
                for region in range(SPATIAL_REGIONS)
            ]
        )
    )
    color = config.color_joint_weight * joint_color + config.color_spatial_weight * spatial_color

    lbp = chi_square_distance(query.texture[:LBP_DIM], candidate.texture[:LBP_DIM])
    glcm = euclidean_distance(
        query.texture[LBP_DIM:BASE_TEXTURE_DIM], candidate.texture[LBP_DIM:BASE_TEXTURE_DIM]
    )
    gabor = euclidean_distance(
        query.texture[BASE_TEXTURE_DIM:], candidate.texture[BASE_TEXTURE_DIM:]
    )
    texture = (
        config.texture_lbp_weight * lbp
        + config.texture_glcm_weight * glcm
        + config.texture_gabor_weight * gabor
    )

    silhouette = euclidean_distance(
        query.shape[:SILHOUETTE_DIM], candidate.shape[:SILHOUETTE_DIM]
    )
    scalars = euclidean_distance(
        query.shape[SILHOUETTE_DIM:GEOMETRY_DIM],
        candidate.shape[SILHOUETTE_DIM:GEOMETRY_DIM],
    )
    hog = euclidean_distance(query.shape[GEOMETRY_DIM:], candidate.shape[GEOMETRY_DIM:])
    shape = (
        config.shape_silhouette_weight * silhouette
        + config.shape_scalar_weight * scalars
        + config.shape_hog_weight * hog
    )
    return GroupDistances(float(color), float(texture), float(shape))


def effective_weights(mask_quality: float, settings: Settings | None = None) -> dict[str, float]:
    """Reduce shape influence for a low-quality query mask, then renormalize."""

    config = settings or get_settings()
    weights = np.asarray(
        [config.weight_color, config.weight_texture, config.weight_shape * np.clip(mask_quality, 0.0, 1.0)],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= EPSILON:
        weights[:] = 1.0 / 3.0
    else:
        weights /= total
    return {"color": float(weights[0]), "texture": float(weights[1]), "shape": float(weights[2])}


def similarities(
    distances: GroupDistances,
    scaling: DistanceScaling,
    weights: dict[str, float],
) -> tuple[float, float, float, float]:
    normalized = scaling.normalize(distances)
    final_distance = (
        weights["color"] * normalized.color
        + weights["texture"] * normalized.texture
        + weights["shape"] * normalized.shape
    )
    return (
        float(np.clip(1.0 - final_distance, 0.0, 1.0)),
        float(np.clip(1.0 - normalized.color, 0.0, 1.0)),
        float(np.clip(1.0 - normalized.texture, 0.0, 1.0)),
        float(np.clip(1.0 - normalized.shape, 0.0, 1.0)),
    )


def fit_distance_scaling(
    bundles: list[FeatureBundle],
    *,
    settings: Settings | None = None,
) -> DistanceScaling:
    """Estimate p5/p95 from deterministic random dataset pairs."""

    config = settings or get_settings()
    if len(bundles) < 2:
        raise ValueError("At least two dataset images are required for distance scaling")
    rng = np.random.default_rng(config.random_seed)
    possible = len(bundles) * (len(bundles) - 1) // 2
    sample_count = min(config.distance_sample_pairs, possible)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < sample_count:
        first, second = rng.integers(0, len(bundles), size=2)
        if first != second:
            pairs.add((min(int(first), int(second)), max(int(first), int(second))))
    values = np.asarray(
        [list(asdict(component_distances(bundles[first], bundles[second], config)).values()) for first, second in pairs],
        dtype=np.float64,
    )
    ranges = [
        DistanceRange(float(np.percentile(values[:, index], 5)), float(np.percentile(values[:, index], 95)))
        for index in range(3)
    ]
    return DistanceScaling(*ranges)
