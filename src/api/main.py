import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from src.api.config import ALLOWED_FORMATS, MAX_UPLOAD_BYTES
from src.api.schemas import (
    HealthResponse,
    PredictResponse,
    Top1,
)
from src.models.inference import VisionXPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configured_path = os.getenv("MODEL_BUNDLE_DIR", "models/test")
    bundle_dir = Path(configured_path).expanduser()

    if not bundle_dir.is_absolute():
        bundle_dir = PROJECT_ROOT / bundle_dir

    bundle_dir = bundle_dir.resolve()

    for filename in ("model.pth", "config.json"):
        if not (bundle_dir / filename).is_file():
            raise FileNotFoundError(f"Missing {filename} in model bundle: {bundle_dir}")

    print(f"Loading model bundle: {bundle_dir}")
    app.state.predictor = VisionXPredictor(bundle_dir, device="cpu")

    try:
        yield
    finally:
        del app.state.predictor


app = FastAPI(
    title="VisionX CXR-CAD API",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=True,
        mode="model",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        image = Image.open(io.BytesIO(data))

        if image.format not in ALLOWED_FORMATS:
            raise HTTPException(status_code=415, detail="Only PNG and JPEG are supported.")

        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    if image.format not in ALLOWED_FORMATS:
        raise HTTPException(status_code=415, detail="Only PNG and JPEG are supported.")

    result = await run_in_threadpool(
        app.state.predictor.analyze,
        image,
    )
    scores = result["scores"]
    top_label = result["top_label"]

    return PredictResponse(
        model_version=result["model_version"],
        is_stub=False,
        scores=scores,
        detected_diseases=[],
        top1=Top1(
            name=top_label,
            score=scores[top_label],
            detected=False,
        ),
        heatmap_base64=result["gradcam_overlay_png_base64"],
        inference_time_ms=(time.perf_counter() - start) * 1000,
    )
