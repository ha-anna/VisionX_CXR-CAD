import numpy as np
import pytest
import torch
from PIL import Image

from src.models.inference import preprocess_tensor


def preprocessing(apply_before_resize):
    """Same settings as the deployed bundles (models/best and the ensemble export)."""
    return {
        "resize_strategy": "direct_resize",
        "grayscale": True,
        "channels": "repeat_grayscale_3_times",
        "resize": [224, 224],
        "interpolation": "INTER_AREA",
        "clahe": True,
        "clahe_settings": {
            "clip_limit": 2.0,
            "tile_grid_size": [8, 8],
            "apply_before_resize": apply_before_resize,
        },
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


@pytest.fixture
def synthetic_xray():
    """Random non-square grayscale image; no real patient data."""
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, size=(300, 260), dtype=np.uint8), mode="L")


@pytest.mark.parametrize("apply_before_resize", [True, False])
def test_output_shape_and_finite(synthetic_xray, apply_before_resize):
    tensor = preprocess_tensor(synthetic_xray, preprocessing(apply_before_resize))

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_channels_are_identical(synthetic_xray):
    tensor = preprocess_tensor(synthetic_xray, preprocessing(True))
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    unnormalized = tensor * std + mean

    torch.testing.assert_close(unnormalized[0], unnormalized[1])
    torch.testing.assert_close(unnormalized[0], unnormalized[2])


def test_deterministic(synthetic_xray):
    first = preprocess_tensor(synthetic_xray, preprocessing(True))
    second = preprocess_tensor(synthetic_xray, preprocessing(True))

    assert torch.equal(first, second)


def test_rgb_input_gives_same_result_as_grayscale(synthetic_xray):
    rgb = synthetic_xray.convert("RGB")

    assert torch.equal(
        preprocess_tensor(rgb, preprocessing(True)),
        preprocess_tensor(synthetic_xray, preprocessing(True)),
    )
