"""API contract tests against a tiny ensemble bundle with random weights.

The bundle has one randomly initialised DenseNet-121 instead of the 10 trained checkpoints,
so the tests check the response format, not model quality.
"""

import base64
import io
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.api.schemas import LABELS, HealthResponse, PredictResponse
from src.models.inference import build_model


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    bundle = tmp_path_factory.mktemp("ensemble_bundle")
    (bundle / "models").mkdir()
    torch.manual_seed(0)
    model = build_model("densenet121", len(LABELS))
    torch.save(model.state_dict(), bundle / "models" / "densenet121_g0_f0.pth")
    config = {
        "archs": ["densenet121"],
        "class_names": LABELS,
        "model_files": ["densenet121_g0_f0.pth"],
        "use_tta": True,
        "temperatures": {label: 1.0 for label in LABELS},
        "preprocessing": {
            "clahe": {"clipLimit": 2.0, "tileGridSize": [8, 8], "applied_before_resize": True},
            "resize": [224, 224],
            "interpolation": "INTER_AREA",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
    }
    (bundle / "inference_config.json").write_text(json.dumps(config), encoding="utf-8")
    return bundle


@pytest.fixture(scope="module")
def client(bundle_dir):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_BUNDLE_DIR", str(bundle_dir))
        from src.api.main import app

        with TestClient(app) as test_client:
            yield test_client


def png_bytes(size=(256, 256)):
    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 256, size=size, dtype=np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_returns_schema_fields(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == set(HealthResponse.model_fields)
    HealthResponse.model_validate(body)
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_schema_fields(client):
    response = client.post("/predict", files={"file": ("xray.png", png_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == set(PredictResponse.model_fields)
    result = PredictResponse.model_validate(body)

    assert list(result.scores) == LABELS
    assert all(0.0 <= score <= 1.0 for score in result.scores.values())
    assert result.top1.name == max(result.scores, key=result.scores.get)
    assert result.top1.score == result.scores[result.top1.name]
    assert result.is_stub is False
    assert result.inference_time_ms > 0
    assert result.heatmap_base64 is not None
    with Image.open(io.BytesIO(base64.b64decode(result.heatmap_base64))) as heatmap:
        assert heatmap.format == "PNG"


def test_predict_rejects_non_image(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})

    assert response.status_code == 400
