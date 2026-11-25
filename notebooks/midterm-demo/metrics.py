"""Метрики: AUROC / AUPRC по каждой болезни, ECE."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import config as C


def per_class_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    out = []
    for k in range(y_true.shape[1]):
        out.append(roc_auc_score(y_true[:, k], y_prob[:, k])
                   if 0 < y_true[:, k].sum() < len(y_true) else np.nan)
    return np.array(out)


def per_class_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    out = []
    for k in range(y_true.shape[1]):
        out.append(average_precision_score(y_true[:, k], y_prob[:, k])
                   if y_true[:, k].sum() > 0 else np.nan)
    return np.array(out)


def ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error для бинарного прогноза."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def mean_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    return float(np.mean([ece_binary(y_true[:, k], y_prob[:, k], n_bins)
                          for k in range(y_true.shape[1])]))


def summarize(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    au, ap = per_class_auroc(y_true, y_prob), per_class_auprc(y_true, y_prob)
    return {
        "mean_auroc": float(np.nanmean(au)), "mean_auprc": float(np.nanmean(ap)),
        "ece": mean_ece(y_true, y_prob),
        "auroc_per_class": dict(zip(C.DISEASES, au.round(4).tolist())),
        "auprc_per_class": dict(zip(C.DISEASES, ap.round(4).tolist())),
    }
