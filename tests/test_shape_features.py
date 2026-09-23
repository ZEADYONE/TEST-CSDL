import numpy as np

from app.services.feature_extractor import BirdFeatureExtractor, SHAPE_DIM
from app.services.hog_features import HOG_DIM
from app.services.shape_features import extract_shape_features


def test_shape_dimensions_and_grid_range(bird_sample):
    _, mask = bird_sample
    vector = extract_shape_features(mask)
    assert vector.shape == (69,)
    assert np.all(np.isfinite(vector))
    assert np.all((vector[:64] >= 0.0) & (vector[:64] <= 1.0))


def test_combined_shape_vector_includes_hog(bird_sample):
    image, mask = bird_sample
    vector = BirdFeatureExtractor().extract(image, mask).shape
    assert vector.shape == (69 + HOG_DIM,)
    assert vector.shape == (SHAPE_DIM,)
