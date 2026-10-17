"""Check that handwriting views retain boundary strokes and orientation."""

import numpy as np
import pytest
import torch
from PIL import Image

from benchmarks.dataset import get_config
from benchmarks.models.simclr import create_transforms


@pytest.mark.parametrize("name,size", [("kannadamnist", 28), ("devanagari", 32)])
def test_handwriting_views_preserve_strokes(name, size):
    cfg = get_config(name)
    train, val, collate = create_transforms(cfg)
    pixels = np.zeros((size, size), dtype=np.uint8)
    pixels[:4, :4] = pixels[-4:, :4] = 255
    pixels[:4, -4:] = pixels[-4:, -4:] = 128
    sample = {"image": Image.fromarray(pixels), "label": 3}
    with torch.random.fork_rng():
        torch.manual_seed(42)
        for _ in range(12):
            batch = collate([train(dict(sample))])
            assert len(batch["views"]) == 2
            for view in batch["views"]:
                image = view["image"]
                assert image.shape == (1, 3, 224, 224)
                assert torch.isfinite(image).all()
                assert view["label"].tolist() == [3]
                restored = image[0, 0] * cfg.std[0] + cfg.mean[0]
                # All four boundary markers survive; brighter markers stay left.
                for rows in (slice(0, 112), slice(112, 224)):
                    left = restored[rows, :112]
                    right = restored[rows, 112:]
                    assert left.max() > 0.9
                    assert 0.45 < right.max() < 0.6
            assert not torch.equal(batch["views"][0]["image"], batch["views"][1]["image"])
    assert torch.equal(val(dict(sample))["image"], val(dict(sample))["image"])
