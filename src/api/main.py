import base64
import io
import os
import time
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.api.config import ALLOWED_FORMATS, MAX_UPLOAD_BYTES
from src.api.schemas import (
    LABELS,
    HealthResponse,
    PredictResponse,
    Top1,
)
from src.models.inference import VisionXPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle_dir = os.getenv("MODEL_BUNDLE_DIR", "models/demo")
    app.state.predictor = VisionXPredictor(bundle_dir, device="cpu")
    yield
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
