import numpy as np

from app.services.feature_extractor import BirdFeatureExtractor
from app.services.similarity import (
    DistanceRange,
    DistanceScaling,
    chi_square_distance,
    component_distances,
    effective_weights,
    similarities,
)


def test_identical_features_have_zero_distance_and_high_similarity(bird_sample):
    image, mask = bird_sample
    bundle = BirdFeatureExtractor().extract(image, mask)
    distances = component_distances(bundle, bundle)
    assert distances.color == 0.0
    assert distances.texture == 0.0
    assert distances.shape == 0.0
    scaling = DistanceScaling(
        DistanceRange(0.0, 1.0), DistanceRange(0.0, 1.0), DistanceRange(0.0, 1.0)
    )
    assert similarities(distances, scaling, effective_weights(1.0))[0] == 1.0


def test_chi_square_handles_all_zero_vectors():
    assert chi_square_distance(np.zeros(8), np.zeros(8)) == 0.0


def test_low_mask_quality_reduces_shape_weight():
    high = effective_weights(1.0)
    low = effective_weights(0.2)
    assert low["shape"] < high["shape"]
    assert np.isclose(sum(low.values()), 1.0)


def test_distance_scaling_preserves_order_below_p5():
    distance_range = DistanceRange(p5=0.2, p95=1.0)
    assert distance_range.normalize(0.05) < distance_range.normalize(0.10)
    assert distance_range.normalize(0.0) == 0.0
