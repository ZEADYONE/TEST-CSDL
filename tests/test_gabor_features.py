import numpy as np

from app.services.feature_extractor import BirdFeatureExtractor, TEXTURE_DIM
from app.services.gabor_features import GABOR_DIM, extract_gabor_features


def test_gabor_dimensions_and_finite_values(bird_sample):
    image, mask = bird_sample
    vector = extract_gabor_features(image, mask)
    assert vector.shape == (GABOR_DIM,)
    assert np.all(np.isfinite(vector))
    assert BirdFeatureExtractor().extract(image, mask).texture.shape == (TEXTURE_DIM,)


def test_gabor_ignores_background_pixels(bird_sample):
    image, mask = bird_sample
    changed = image.copy()
    changed[~mask] = (255, 255, 255)
    assert np.allclose(
        extract_gabor_features(image, mask),
        extract_gabor_features(changed, mask),
        atol=1e-6,
    )


def test_gabor_responds_to_texture_orientation(bird_sample):
    image, mask = bird_sample
    horizontal = np.zeros_like(image)
    vertical = np.zeros_like(image)
    horizontal[(np.indices(mask.shape)[0] // 8) % 2 == 0] = 255
    vertical[(np.indices(mask.shape)[1] // 8) % 2 == 0] = 255
    horizontal[~mask] = 0
    vertical[~mask] = 0
    assert np.linalg.norm(
        extract_gabor_features(horizontal, mask) - extract_gabor_features(vertical, mask)
    ) > 0.1
