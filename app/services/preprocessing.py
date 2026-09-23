"""Shared bird segmentation, mask validation, and 384x384 normalization."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import Settings, get_settings
from app.services.orientation import normalize_orientation


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaskQuality:
    valid: bool
    object_ratio: float
    num_components: int
    touches_border: bool
    quality_score: float


@dataclass(frozen=True)
class ProcessedBird:
    original_bgr: np.ndarray
    image: np.ndarray
    mask: np.ndarray
    quality: MaskQuality
    cropped_image: np.ndarray
    cropped_mask: np.ndarray


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    values = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if values.size and float(values.max()) <= 1.0:
        values *= 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded grayscale/BGR/BGRA image bytes without a temporary file."""

    if not data:
        raise ValueError("The uploaded image is empty")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("The uploaded file is not a supported image")
    return _to_uint8(image)


def load_image(path: str | Path) -> np.ndarray:
    """Read an image using imdecode, which supports Unicode paths on Windows."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        encoded = np.fromfile(source, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Cannot read image: {source}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unsupported or corrupt image: {source}")
    return _to_uint8(image)


def bgr_and_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize grayscale/BGR/BGRA input and preserve alpha when present."""

    image = _to_uint8(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
    if image.ndim != 3:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")
    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
    if channels == 3:
        return image, None
    if channels == 4:
        return image[:, :, :3], image[:, :, 3]
    raise ValueError(f"Unsupported channel count: {channels}")


def infer_foreground_mask(image_bgr: np.ndarray, threshold: float = 18.0) -> np.ndarray:
    """Infer foreground against a nearly constant background using corner color.

    This conservative fallback is used only for an image explicitly declared
    already-preprocessed when no alpha channel exists.
    """

    height, width = image_bgr.shape[:2]
    block = max(1, min(height, width) // 20)
    corners = np.concatenate(
        (
            image_bgr[:block, :block].reshape(-1, 3),
            image_bgr[:block, -block:].reshape(-1, 3),
            image_bgr[-block:, :block].reshape(-1, 3),
            image_bgr[-block:, -block:].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(corners.astype(np.float32), axis=0)
    distance = np.linalg.norm(image_bgr.astype(np.float32) - background, axis=2)
    return distance > threshold


def clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply light opening/closing without aggressive contour erosion."""

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    size = max(1, int(kernel_size))
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed.astype(bool)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return a half-open ``(x1, y1, x2, y2)`` bounding box."""

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def validate_mask(mask: np.ndarray, settings: Settings | None = None) -> MaskQuality:
    """Measure mask usability and return a normalized, explainable score."""

    config = settings or get_settings()
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or binary.size == 0:
        return MaskQuality(False, 0.0, 0, False, 0.0)
    height, width = binary.shape
    area = int(binary.sum())
    object_ratio = area / float(height * width)
    bbox = mask_bbox(binary)

    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    component_areas = component_stats[1:, cv2.CC_STAT_AREA] if component_count > 1 else []
    significant_cutoff = max(4, int(area * 0.001))
    num_components = int(sum(int(value) >= significant_cutoff for value in component_areas))

    edge_fraction = max(
        float(binary[0].mean()),
        float(binary[-1].mean()),
        float(binary[:, 0].mean()),
        float(binary[:, -1].mean()),
    )
    touches_border = edge_fraction > config.mask_border_fraction

    dimensions_valid = False
    bbox_score = 0.0
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        min_dimension = min(x2 - x1, y2 - y1)
        dimensions_valid = min_dimension >= config.mask_min_dimension
        bbox_score = min(1.0, min_dimension / max(1.0, config.mask_min_dimension * 2.0))

    area_score = min(1.0, object_ratio / max(config.mask_min_ratio * 2.0, 1e-8))
    component_score = 1.0 if num_components == 1 else max(0.25, 1.0 / max(1, num_components))
    border_score = 0.65 if touches_border else 1.0
    score = float(np.clip(0.45 * area_score + 0.25 * bbox_score + 0.20 * component_score + 0.10 * border_score, 0.0, 1.0))
    valid = bool(area > 0 and object_ratio >= config.mask_min_ratio and dimensions_valid)
    return MaskQuality(valid, object_ratio, num_components, touches_border, score if valid else 0.0)


def _padded_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], padding: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    height, width = shape
    pad_x = int(np.ceil((x2 - x1) * padding))
    pad_y = int(np.ceil((y2 - y1) * padding))
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def normalize_bird(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    settings: Settings | None = None,
    flip_horizontal: bool = False,
) -> ProcessedBird:
    """Crop a masked bird, resize with aspect ratio, and center on a square canvas."""

    config = settings or get_settings()
    if image_bgr.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask sizes differ")
    cleaned = clean_mask(mask, config.morphology_kernel)
    quality = validate_mask(cleaned, config)
    if not quality.valid:
        raise ValueError(
            f"Invalid bird mask (ratio={quality.object_ratio:.4f}, components={quality.num_components})"
        )
    bbox = mask_bbox(cleaned)
    assert bbox is not None
    x1, y1, x2, y2 = _padded_bbox(bbox, cleaned.shape, config.bbox_padding)
    crop_image = image_bgr[y1:y2, x1:x2].copy()
    crop_mask = cleaned[y1:y2, x1:x2].copy()
    height, width = crop_mask.shape
    scale = min(config.target_size / width, config.target_size / height)
    new_width = max(1, min(config.target_size, round(width * scale)))
    new_height = max(1, min(config.target_size, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized_image = cv2.resize(crop_image, (new_width, new_height), interpolation=interpolation)
    resized_mask = cv2.resize(
        crop_mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    canvas = np.zeros((config.target_size, config.target_size, 3), dtype=np.uint8)
    canvas_mask = np.zeros((config.target_size, config.target_size), dtype=bool)
    offset_x = (config.target_size - new_width) // 2
    offset_y = (config.target_size - new_height) // 2
    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized_image
    canvas_mask[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized_mask
    canvas[~canvas_mask] = 0
    canvas, canvas_mask = normalize_orientation(
        canvas,
        canvas_mask,
        mode=config.orientation_mode,
        flip_horizontal=flip_horizontal,
    )
    return ProcessedBird(image_bgr.copy(), canvas, canvas_mask, quality, crop_image, crop_mask)


class YoloBirdSegmenter:
    """Lazily loaded YOLO-Seg adapter that selects the best bird instance."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is None:
            if not self.settings.yolo_model.is_file():
                raise FileNotFoundError(
                    f"YOLO model not found: {self.settings.yolo_model}. Set YOLO_MODEL in .env."
                )
            from ultralytics import YOLO

            self._model = YOLO(str(self.settings.yolo_model))
        return self._model

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        # The shared Ultralytics predictor has mutable state, so serialize its
        # inference while still avoiding a costly model reload per query.
        with self._lock:
            result = self._load().predict(
                source=image_bgr,
                conf=0.001,
                retina_masks=True,
                verbose=False,
            )[0]
        if result.boxes is None or result.masks is None or result.masks.data is None:
            raise ValueError("YOLO-Seg did not detect a bird")
        names = getattr(result, "names", {})
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
        raw_masks = result.masks.data.detach().cpu().numpy()
        height, width = image_bgr.shape[:2]
        candidates: list[tuple[float, np.ndarray]] = []
        for class_id, confidence, raw_mask in zip(classes, confidences, raw_masks):
            name = names.get(int(class_id), class_id) if isinstance(names, dict) else names[int(class_id)]
            if str(name).lower() != "bird" or confidence < self.settings.yolo_confidence:
                continue
            mask = np.squeeze(raw_mask).astype(np.float32)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
            binary = mask > 0.5
            area_ratio = float(binary.mean())
            score = float(confidence) * np.sqrt(max(area_ratio, 1e-8))
            candidates.append((score, binary))
        if not candidates:
            raise ValueError("YOLO-Seg did not detect a bird above the confidence threshold")
        return max(candidates, key=lambda item: item[0])[1]


@lru_cache(maxsize=1)
def get_yolo_segmenter() -> YoloBirdSegmenter:
    """Return the lazily loaded, process-wide query segmenter."""

    return YoloBirdSegmenter(get_settings())


def preprocess_array(
    image: np.ndarray,
    *,
    already_preprocessed: bool,
    flip_horizontal: bool = False,
    segmenter: YoloBirdSegmenter | None = None,
    settings: Settings | None = None,
) -> ProcessedBird:
    """Run the appropriate dataset/query branch of the shared pipeline."""

    config = settings or get_settings()
    bgr, alpha = bgr_and_alpha(image)
    if already_preprocessed:
        mask = alpha > 0 if alpha is not None else infer_foreground_mask(bgr)
    else:
        mask = (segmenter or get_yolo_segmenter()).segment(bgr)
    return normalize_bird(
        bgr,
        mask,
        settings=config,
        flip_horizontal=flip_horizontal,
    )


def rgba_image(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Build an OpenCV BGRA image for transparent PNG output."""

    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask.astype(np.uint8) * 255
    return bgra


def write_image(path: Path, image: np.ndarray) -> None:
    """Unicode-safe image writer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    encoded.tofile(path)
