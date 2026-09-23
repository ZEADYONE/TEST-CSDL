import numpy as np
import cv2

from app.services.color_features import (
    COLOR_DIM,
    JOINT_DIM,
    SPATIAL_REGION_DIM,
    SPATIAL_REGIONS,
    extract_color_features,
)
from app.services.similarity import chi_square_distance


def test_color_dimensions_and_histogram_normalization(bird_sample):
    image, mask = bird_sample
    vector = extract_color_features(image, mask)
    assert vector.shape == (COLOR_DIM,)
    assert np.all(np.isfinite(vector))
    assert np.isclose(vector[:JOINT_DIM].sum(), 1.0)
    for region in range(SPATIAL_REGIONS):
        histogram = vector[
            JOINT_DIM
            + region * SPATIAL_REGION_DIM : JOINT_DIM
            + (region + 1) * SPATIAL_REGION_DIM
        ]
        assert np.isclose(histogram.sum(), 1.0)


def test_background_pixels_do_not_change_color_features(bird_sample):
    image, mask = bird_sample
    changed = image.copy()
    changed[~mask] = (0, 255, 0)
    assert np.allclose(extract_color_features(image, mask), extract_color_features(changed, mask))


def test_fine_circular_hue_separates_red_from_yellow():
    mask = np.ones((64, 64), dtype=bool)
    red = np.full((64, 64, 3), (0, 0, 255), dtype=np.uint8)
    yellow = np.full((64, 64, 3), (0, 255, 255), dtype=np.uint8)
    wrapped_hsv = np.full((64, 64, 3), (179, 255, 255), dtype=np.uint8)
    wrapped_red = cv2.cvtColor(wrapped_hsv, cv2.COLOR_HSV2BGR)
    red_features = extract_color_features(red, mask)[:JOINT_DIM]
    yellow_features = extract_color_features(yellow, mask)[:JOINT_DIM]
    wrapped_features = extract_color_features(wrapped_red, mask)[:JOINT_DIM]
    assert chi_square_distance(red_features, wrapped_features) < chi_square_distance(
        red_features, yellow_features
    )
