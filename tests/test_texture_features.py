import numpy as np

from app.services.texture_features import extract_texture_features


def test_texture_dimensions_and_finite_values(bird_sample):
    image, mask = bird_sample
    vector = extract_texture_features(image, mask)
    assert vector.shape == (38,)
    assert np.all(np.isfinite(vector))
    assert np.isclose(vector[:10].sum(), 1.0)
    assert np.isclose(vector[10:28].sum(), 1.0)


def test_background_pixels_do_not_change_texture_features(bird_sample):
    image, mask = bird_sample
    changed = image.copy()
    changed[~mask] = (255, 255, 255)
    assert np.allclose(
        extract_texture_features(image, mask),
        extract_texture_features(changed, mask),
        atol=1e-6,
    )
