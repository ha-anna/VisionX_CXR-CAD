import base64
import io
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.api.config import ALLOWED_FORMATS, MAX_UPLOAD_BYTES
from src.api.schemas import (
    LABELS,
    HealthResponse,
    PredictResponse,
    Top1,
)

app = FastAPI(title="VisionX CXR-CAD API")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=False, mode="stub")


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # actually decodes the pixels, catches corrupt files
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    if image.format not in ALLOWED_FORMATS:
        raise HTTPException(status_code=415, detail="Only PNG and JPEG are supported.")

    # STUB: fixed fake scores. Not real inference.
    scores = {label: 0.05 for label in LABELS}
    scores["Effusion"] = 0.42

    # 1x1 placeholder PNG, standing in for a heatmap
    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="PNG")
    heatmap = base64.b64encode(buf.getvalue()).decode()

    return PredictResponse(
        is_stub=True,
        scores=scores,
        detected_diseases=[],
        top1=Top1(name="Effusion", score=0.42, detected=False),
        heatmap_base64=heatmap,
        inference_time_ms=(time.perf_counter() - start) * 1000,
    )
