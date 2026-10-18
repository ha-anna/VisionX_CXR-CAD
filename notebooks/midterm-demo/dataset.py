"""Dataset для ChestX-ray14: читает уменьшенные снимки из images_224.npy (memmap)."""
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config as C

cv2.setNumThreads(0)  # без этого каждый воркер плодит потоки OpenCV и они мешают друг другу


def load_metadata(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or C.METADATA_CSV)
    df["row"] = np.arange(len(df))  # позиция строки в images_224.npy
    return df


def get_split(df: pd.DataFrame, fold: int | None = None, part: str = "train",
              only_quality_ok: bool = True) -> pd.DataFrame:
    """part='train'|'val' (внутри Train, по фолду) или 'test' (фиксированный Test)."""
    if part == "test":
        out = df[df.split == "test"]
    else:
        tr = df[df.split == "train"]
        out = tr[tr.fold != fold] if part == "train" else tr[tr.fold == fold]
    if only_quality_ok:
        out = out[out.quality_ok]
    return out.reset_index(drop=True)


def apply_clahe(img_u8: np.ndarray, clip=C.CLAHE_CLIP, tile=C.CLAHE_TILE, _cache={}) -> np.ndarray:
    key = (clip, tile)
    if key not in _cache:  # объект создаём лениво (в каждом процессе свой)
        _cache[key] = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return _cache[key].apply(img_u8)


class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = False, use_clahe: bool = True,
                 flip_aug: bool = True, npy_path=None):
        self.rows = df["row"].values
        self.labels = df[C.DISEASES].values.astype(np.float32)
        self.train = train
        self.use_clahe = use_clahe
        self.flip_aug = flip_aug
        self.npy_path = str(npy_path or C.IMAGES_NPY)
        self._arr = None
        self.mean = np.array(C.IMAGENET_MEAN, np.float32)[:, None, None]
        self.std = np.array(C.IMAGENET_STD, np.float32)[:, None, None]

    def __len__(self):
        return len(self.rows)

    def _images(self):
        if self._arr is None:  # открываем memmap внутри воркера
            self._arr = np.load(self.npy_path, mmap_mode="r")
        return self._arr

    def _augment(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        angle = np.random.uniform(-10, 10)
        scale = np.random.uniform(0.9, 1.1)
        tx, ty = np.random.uniform(-0.05, 0.05, 2) * w
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[:, 2] += (tx, ty)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        if self.flip_aug and np.random.rand() < 0.5:
            img = img[:, ::-1]
        return np.ascontiguousarray(img)

    def __getitem__(self, i):
        img = np.array(self._images()[self.rows[i]])  # копия uint8 (H, W)
        if self.train:
            img = self._augment(img)
        if self.use_clahe:
            img = apply_clahe(img)
        x = img.astype(np.float32) / 255.0
        x = np.repeat(x[None], 3, axis=0)  # 1 канал -> 3 канала
        x = (x - self.mean) / self.std
        return torch.from_numpy(x), torch.from_numpy(self.labels[i])
