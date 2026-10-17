"""Central application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Validated settings shared by web, indexing, and feature services."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    app_name: str = "Bird Image Retrieval"
    database_url: str = "postgresql+psycopg://postgres:password@localhost:5432/bird_cbir"
    yolo_model: Path = PROJECT_ROOT / "models" / "yolo11l-seg.pt"
    target_size: int = 384
    mask_min_ratio: float = 0.01
    mask_min_dimension: int = 8
    mask_border_fraction: float = 0.35
    bbox_padding: float = 0.10
    morphology_kernel: int = 3
    yolo_confidence: float = 0.05
    orientation_mode: str = "none"
    debug_features: bool = False
    feature_version: str = "v5-bbox-shape"

    color_joint_weight: float = 0.30
    color_spatial_weight: float = 0.70
    texture_lbp_weight: float = 0.10
    texture_glcm_weight: float = 0.30
    texture_gabor_weight: float = 0.60
    shape_silhouette_weight: float = 0.80
    shape_hog_weight: float = 0.20
    weight_color: float = 0.30
    weight_texture: float = 0.30
    weight_shape: float = 0.40

    scaler_dir: Path = PROJECT_ROOT / "models"
    distance_stats_path: Path = PROJECT_ROOT / "models" / "distance_stats.json"
    static_dir: Path = PROJECT_ROOT / "app" / "static"
    debug_dir: Path = PROJECT_ROOT / "debug_outputs"
    max_upload_mb: int = 15
    random_seed: int = 42
    distance_sample_pairs: int = 20_000

    @field_validator(
        "target_size",
        "mask_min_dimension",
        "morphology_kernel",
        "max_upload_mb",
    )
    @classmethod
    def positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("yolo_model", "scaler_dir", "distance_stats_path", "static_dir", "debug_dir")
    @classmethod
    def project_relative_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator(
        "mask_min_ratio", "mask_border_fraction", "bbox_padding", "yolo_confidence"
    )
    @classmethod
    def unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("must be in [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_weights(self) -> "Settings":
        pairs = (
            (self.color_joint_weight, self.color_spatial_weight, "color"),
        )
        for first, second, name in pairs:
            if abs(first + second - 1.0) > 1e-6:
                raise ValueError(f"{name} sub-weights must sum to 1")
        if abs(
            self.texture_lbp_weight
            + self.texture_glcm_weight
            + self.texture_gabor_weight
            - 1.0
        ) > 1e-6:
            raise ValueError("texture sub-weights must sum to 1")
        if abs(
            self.shape_silhouette_weight
            + self.shape_hog_weight
            - 1.0
        ) > 1e-6:
            raise ValueError("shape sub-weights must sum to 1")
        if abs(self.weight_color + self.weight_texture + self.weight_shape - 1.0) > 1e-6:
            raise ValueError("final weights must sum to 1")
        if self.orientation_mode not in {"none"}:
            raise ValueError("The current feature pipeline supports ORIENTATION_MODE=none only")
        return self

    def ensure_directories(self) -> None:
        for path in (
            self.scaler_dir,
            self.static_dir / "dataset",
            self.static_dir / "queries",
            self.debug_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable-by-convention configuration."""

    return Settings()
