from pydantic import BaseModel, Field

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


class Detection(BaseModel):
    name: str
    score: float = Field(ge=0, le=1)


class Top1(Detection):
    detected: bool


class PredictResponse(BaseModel):
    schema_version: str = "0.1"
    is_stub: bool
    scores: dict[str, float]
    detected_diseases: list[Detection]
    top1: Top1
    heatmap_base64: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mode: str
