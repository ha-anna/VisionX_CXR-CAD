"""Все настройки проекта. Любое значение можно переопределить переменной окружения,
например:  set CXR_BATCH_SIZE=32  (PowerShell: $env:CXR_BATCH_SIZE=32)."""
import os
from pathlib import Path

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
]


def _env(name, default, cast=str):
    return cast(os.environ.get(f"CXR_{name}", default))


DATA_DIR = Path(_env("DATA_DIR", r"C:\cxr\data"))
RUNS_DIR = Path(_env("RUNS_DIR", r"C:\cxr\runs"))
IMAGES_NPY = DATA_DIR / "images_224.npy"
METADATA_CSV = DATA_DIR / "metadata.csv"

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CLAHE
CLAHE_CLIP = _env("CLAHE_CLIP", 2.0, float)
CLAHE_TILE = (8, 8)

# обучение
BATCH_SIZE = _env("BATCH_SIZE", 64, int)
NUM_WORKERS = _env("NUM_WORKERS", 8, int)
LR = _env("LR", 1e-4, float)
WEIGHT_DECAY = _env("WEIGHT_DECAY", 1e-4, float)
EPOCHS = _env("EPOCHS", 30, int)
PATIENCE = _env("PATIENCE", 5, int)
SEED = _env("SEED", 42, int)
POS_WEIGHT_CAP = _env("POS_WEIGHT_CAP", 30.0, float)
# channels_last на этой связке PyTorch/DenseNet замедляет обучение в 5-7 раз -> выключено
CHANNELS_LAST = _env("CHANNELS_LAST", 0, int)
