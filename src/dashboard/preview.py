import base64
import io
import os
from pathlib import Path

import requests
import streamlit as st
from image_utils import decode_image
from PIL import Image, ImageDraw, ImageFilter
from settings import LABELS

SAMPLE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/a/a1/"
    "Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg"
)
SAMPLE_SOURCE = (
    "https://commons.wikimedia.org/wiki/"
    "File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg"
)


@st.cache_data(show_spinner=False)
def sample_bytes(local_path):
    """Prefer a local sample for offline demos; otherwise fetch the public CC0 image."""
    if local_path:
        return Path(local_path).read_bytes()
    response = requests.get(
        SAMPLE_URL,
        headers={"User-Agent": "VisionX educational UI preview"},
        timeout=(5, 20),
    )
    response.raise_for_status()

    return response.content


def preview_images():
    image = decode_image(sample_bytes(os.getenv("PREVIEW_XRAY_PATH", "")))

    small = image.copy()
    small.thumbnail((1000, 1000))
    width, height = small.size
    layer = Image.new("RGBA", small.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.ellipse(
        (width * 0.42, height * 0.47, width * 0.73, height * 0.76),
        fill=(238, 88, 74, 140),
    )
    layer = layer.filter(ImageFilter.GaussianBlur(radius=width * 0.04))
    overlay = Image.alpha_composite(small.convert("RGBA"), layer).convert("RGB")
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")

    return image, "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def preview_response():
    values = [0.23, 0.89, 0.57, 0.12, 0.09, 0.16, 0.05, 0.03, 0.08, 0.11, 0.07, 0.02, 0.09, 0.01]

    return {
        "predictions": dict(zip(LABELS, values)),
        "detected_diseases": ["Cardiomegaly", "Effusion"],
        "top1_disease": "Cardiomegaly",
        "gradcam_heatmap": None,
    }
