import numpy as np

from app.config import Settings
from app.services.preprocessing import normalize_bird, validate_mask


def test_mask_validation_rejects_empty_mask():
    quality = validate_mask(np.zeros((100, 100), dtype=np.uint8))
    assert not quality.valid
    assert quality.quality_score == 0.0


def test_normalization_outputs_target_size(bird_sample, tmp_path):
    image, mask = bird_sample
    settings = Settings(target_size=384, scaler_dir=tmp_path, distance_stats_path=tmp_path / "d.json")
    result = normalize_bird(image, mask, settings=settings)
    assert result.image.shape == (384, 384, 3)
    assert result.mask.shape == (384, 384)
    assert result.quality.valid
