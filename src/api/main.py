import base64
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from src.api.schemas import (
    HealthResponse,
    PredictResponse,
    Top1,
)
from src.api.uploads import read_upload
from src.api.uptime import get_uptime
from src.models.inference import (
    VisionXPredictor,
    analyze_ensemble,
    ensemble_grad_cam,
    load_ensemble_models,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configured_path = os.getenv("MODEL_BUNDLE_DIR", "models/test")
    bundle_dir = Path(configured_path).expanduser()

    if not bundle_dir.is_absolute():
        bundle_dir = PROJECT_ROOT / bundle_dir

    bundle_dir = bundle_dir.resolve()

    print(f"Loading model bundle: {bundle_dir}")

    if (bundle_dir / "inference_config.json").is_file():
        app.state.predictor = None
        app.state.ensemble = load_ensemble_models(bundle_dir, device="cpu")
    else:
        for filename in ("model.pth", "config.json"):
            if not (bundle_dir / filename).is_file():
                raise FileNotFoundError(f"Missing {filename} in model bundle: {bundle_dir}")

        app.state.predictor = VisionXPredictor(bundle_dir, device="cpu")
        app.state.ensemble = None

    app.state.model_version = bundle_dir.name
    app.state.ensemble_lock = Lock()

    try:
        yield
    finally:
        del app.state.predictor
        del app.state.ensemble
        del app.state.model_version
        del app.state.ensemble_lock


app = FastAPI(
    title="VisionX CXR-CAD API",
    lifespan=lifespan,
)


@app.get("/", response_class=FileResponse, include_in_schema=False)
def root():
    return FileResponse(Path(__file__).with_name("index.html"), media_type="text/html")


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

    predictor = getattr(app.state, "predictor", None)
    ensemble = getattr(app.state, "ensemble", None)
    if predictor is None and ensemble is None:
        raise HTTPException(
            status_code=503,
            detail="The model is currently unavailable.",
        )

    print("[gradcam] decoding upload", flush=True)
    data, image = await read_upload(file)
    print("[gradcam] upload decoded", flush=True)
    if ensemble is not None:
        config, models_by_arch = ensemble

        try:
            scores, top_label, heatmap_base64 = await run_in_threadpool(
                analyze_ensemble,
                data,
                config,
                models_by_arch,
                app.state.ensemble_lock,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="File could not be decoded for inference."
            ) from exc
        except (RuntimeError, OSError) as exc:
            raise HTTPException(
                status_code=503, detail="The model could not process the image."
            ) from exc

        return PredictResponse(
            model_version=app.state.model_version,
            is_stub=False,
            scores=scores,
            detected_diseases=[],
            top1=Top1(name=top_label, score=scores[top_label], detected=False),
            heatmap_base64=heatmap_base64,
            inference_time_ms=(time.perf_counter() - start) * 1000,
        )

    try:
        result = await run_in_threadpool(
            predictor.analyze,
            image,
        )
    except (RuntimeError, OSError):
        raise HTTPException(
            status_code=503,
            detail="The model could not process the image.",
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


@app.post("/gradcam")
async def gradcam(disease: str, file: UploadFile = File(...)):
    predictor = getattr(app.state, "predictor", None)
    ensemble = getattr(app.state, "ensemble", None)

    if ensemble is not None:
        config, models_by_arch = ensemble
        labels = config["class_names"]
    elif predictor is not None:
        labels = predictor.labels
    else:
        raise HTTPException(status_code=503, detail="The model is currently unavailable.")

    if disease not in labels:
        raise HTTPException(status_code=422, detail="Unknown disease label.")

    data, image = await read_upload(file)

    try:
        if ensemble is not None:
            explanation = await run_in_threadpool(
                ensemble_grad_cam,
                data,
                config,
                models_by_arch,
                app.state.ensemble_lock,
                disease,
            )
        else:
            explanation = await run_in_threadpool(predictor.grad_cam, image, disease)

        print("[gradcam] overlay returned", flush=True)
        if explanation["empty"]:
            return {"disease": disease, "heatmap_base64": None}

        buffer = io.BytesIO()
        explanation["overlay"].save(buffer, format="PNG")
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Grad-CAM generation failed.") from exc

    return {
        "disease": disease,
        "heatmap_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


@app.get("/uptime")
def uptime():
    try:
        return get_uptime()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Uptime history is temporarily unavailable.",
        ) from exc
