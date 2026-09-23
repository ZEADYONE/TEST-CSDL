"""Auditable visual outputs for academic feature inspection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.preprocessing import ProcessedBird, mask_bbox, rgba_image, write_image


def save_debug_outputs(processed: ProcessedBird, output_dir: Path) -> None:
    """Save source, crop, mask, normalized image, grid, and spatial regions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_image(output_dir / "01_original.png", processed.original_bgr)
    write_image(output_dir / "02_mask.png", processed.mask.astype(np.uint8) * 255)
    write_image(output_dir / "03_cropped.png", processed.cropped_image)
    write_image(output_dir / "04_processed_384.png", rgba_image(processed.image, processed.mask))

    silhouette = cv2.cvtColor(processed.mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    height, width = processed.mask.shape
    for index in range(1, 8):
        cv2.line(silhouette, (index * width // 8, 0), (index * width // 8, height), (0, 90, 255), 1)
        cv2.line(silhouette, (0, index * height // 8), (width, index * height // 8), (0, 90, 255), 1)
    write_image(output_dir / "05_silhouette_grid.png", silhouette)

    spatial = processed.image.copy()
    bbox = mask_bbox(processed.mask)
    if bbox:
        x1, y1, x2, y2 = bbox
        boundaries = np.linspace(x1, x2, 4, dtype=int)
        colors = ((65, 179, 255), (109, 227, 201), (199, 128, 255))
        overlay = spatial.copy()
        for index in range(3):
            region = processed.mask.copy()
            region[:, : boundaries[index]] = False
            region[:, boundaries[index + 1] :] = False
            overlay[region] = colors[index]
        spatial = cv2.addWeighted(spatial, 0.55, overlay, 0.45, 0)
        for boundary in boundaries[1:-1]:
            cv2.line(spatial, (int(boundary), y1), (int(boundary), y2), (255, 255, 255), 2)
    write_image(output_dir / "06_spatial_regions.png", spatial)
