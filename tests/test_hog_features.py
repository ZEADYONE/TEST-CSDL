import numpy as np

from app.services.hog_features import HOG_DIM, HOG_REGION_DIM, extract_hog_features


def test_hog_dimensions_and_finite_values(bird_sample):
    image, mask = bird_sample
    vector = extract_hog_features(image, mask)
    assert vector.shape == (HOG_DIM,)
    assert np.all(np.isfinite(vector))

    regions = vector.reshape(-1, HOG_REGION_DIM)
    assert regions.shape == (49, 36)
    assert np.all(np.linalg.norm(regions, axis=1) <= 1.0001)


def test_hog_ignores_background_pixels(bird_sample):
    image, mask = bird_sample
    changed = image.copy()
    changed[~mask] = (5, 240, 60)
    assert np.allclose(
        extract_hog_features(image, mask),
        extract_hog_features(changed, mask),
    )


def test_hog_responds_to_foreground_structure(bird_sample):
    image, mask = bird_sample
    smoothed = image.copy()
    smoothed[mask] = (120, 120, 120)
    distance = np.linalg.norm(
        extract_hog_features(image, mask) - extract_hog_features(smoothed, mask)
    )
    assert distance > 0.1
