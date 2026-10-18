"""
Предобработка NIH ChestX-ray14 для проекта CXR-CAD.

Что делает:
  1. Читает картинки НАПРЯМУЮ из zip (или из распакованной папки) - распаковывать 42 ГБ не нужно.
  2. Считает среднюю яркость каждого снимка и помечает слишком тёмные/светлые (quality_ok).
  3. Уменьшает до SIZE x SIZE (grayscale, uint8) и пишет в один файл images_<SIZE>.npy.
  4. Строит multi-hot метки для 14 болезней + отдельный флаг no_finding.
  5. Делит по Patient ID: Test 15% (фиксированный) и 5-fold GroupKFold внутри Train (85%).
  6. Проверяет отсутствие утечки пациентов между наборами.
  7. Считает prevalence, pos_weight (по Train) и co-occurrence матрицу (heatmap).

Запуск (пример для Windows):
  python prep_data.py --src "D:\\Downloads\\archive.zip" --out "D:\\cxr\\data"
"""
import argparse
import json
import os
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from tqdm import tqdm

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
]
SEED = 42

# ---------- чтение файлов ----------
_ZF = None  # zip-хэндл внутри каждого воркера


def _init_worker(zip_path):
    global _ZF
    _ZF = zipfile.ZipFile(zip_path) if zip_path else None


def _load_one(args):
    """Возвращает (i, уменьшенная картинка uint8, средняя яркость исходника, ok)."""
    i, ref, size = args
    try:
        data = _ZF.read(ref) if _ZF is not None else open(ref, "rb").read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("decode failed")
        mean = float(img.mean())
        small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return i, small, mean, True
    except Exception as e:  # битый файл не должен ронять весь прогон
        print(f"[WARN] {ref}: {e}")
        return i, np.zeros((size, size), np.uint8), -1.0, False


def build_file_index(src):
    """basename -> (путь в zip | путь на диске). Возвращает (index, csv_bytes_or_path, zip_path)."""
    index, csv_ref, zip_path = {}, None, None
    if src.lower().endswith(".zip"):
        zip_path = src
        with zipfile.ZipFile(src) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if base == "Data_Entry_2017.csv":
                    csv_ref = zf.read(name)
                elif base.lower().endswith(".png"):
                    index[base] = name
    else:
        for root, _, files in os.walk(src):
            for f in files:
                p = os.path.join(root, f)
                if f == "Data_Entry_2017.csv":
                    csv_ref = open(p, "rb").read()
                elif f.lower().endswith(".png"):
                    index[f] = p
    if csv_ref is None:
        raise FileNotFoundError("Data_Entry_2017.csv не найден в источнике")
    return index, csv_ref, zip_path


# ---------- метаданные ----------
def build_metadata(csv_bytes):
    import io
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df = df.rename(columns={
        "Image Index": "image", "Finding Labels": "labels", "Patient ID": "patient_id",
        "Patient Age": "age", "Patient Gender": "sex", "View Position": "view",
    })[["image", "labels", "patient_id", "age", "sex", "view"]]
    # в датасете встречаются невозможные возраста (например 148, 414)
    df.loc[(df["age"] < 0) | (df["age"] > 110), "age"] = np.nan
    for d in DISEASES:
        df[d] = df["labels"].apply(lambda s: int(d in s.split("|")))
    df["no_finding"] = (df["labels"] == "No Finding").astype(int)  # отдельно, не 15-й класс
    df["age_group"] = pd.cut(df["age"], [-1, 40, 60, 200], labels=["0-40", "40-60", "60+"],
                             right=False).astype(object)
    return df


def make_splits(df, test_size=0.15, n_folds=5):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr_idx, te_idx = next(gss.split(df, groups=df["patient_id"]))
    df["split"] = "train"
    df.loc[df.index[te_idx], "split"] = "test"
    df["fold"] = -1
    train = df[df["split"] == "train"]
    gkf = GroupKFold(n_splits=n_folds)
    for k, (_, va) in enumerate(gkf.split(train, groups=train["patient_id"])):
        df.loc[train.index[va], "fold"] = k
    return df


def leakage_check(df, n_folds=5):
    lines, ok = [], True
    test_p = set(df[df.split == "test"].patient_id)
    train_p = set(df[df.split == "train"].patient_id)
    inter = train_p & test_p
    ok &= len(inter) == 0
    lines.append(f"Train patients: {len(train_p):,} | Test patients: {len(test_p):,}")
    lines.append(f"Train ∩ Test: {len(inter)} [{'PASS' if not inter else 'FAIL'}]")
    for a in range(n_folds):
        for b in range(a + 1, n_folds):
            pa = set(df[df.fold == a].patient_id)
            pb = set(df[df.fold == b].patient_id)
            n = len(pa & pb)
            ok &= n == 0
            lines.append(f"Fold {a} ∩ Fold {b}: {n} [{'PASS' if n == 0 else 'FAIL'}]")
    lines.append("=> Утечки нет" if ok else "=> УТЕЧКА ОБНАРУЖЕНА!")
    return ok, "\n".join(lines)


def prevalence_table(df):
    train = df[(df.split == "train") & df.quality_ok]
    rows = []
    for d in DISEASES:
        pos = int(train[d].sum())
        neg = len(train) - pos
        rows.append({"disease": d, "count_all": int(df[d].sum()),
                     "prevalence_all_%": round(100 * df[d].mean(), 2),
                     "count_train": pos, "pos_weight": round(neg / max(pos, 1), 2)})
    rows.append({"disease": "No Finding", "count_all": int(df.no_finding.sum()),
                 "prevalence_all_%": round(100 * df.no_finding.mean(), 2),
                 "count_train": int(train.no_finding.sum()), "pos_weight": np.nan})
    return pd.DataFrame(rows)


def cooccurrence_plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = df[DISEASES].values.T @ df[DISEASES].values
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.log1p(m), cmap="viridis")
    ax.set_xticks(range(14)); ax.set_yticks(range(14))
    ax.set_xticklabels(DISEASES, rotation=90); ax.set_yticklabels(DISEASES)
    for i in range(14):
        for j in range(14):
            ax.text(j, i, f"{m[i, j]}", ha="center", va="center", color="w", fontsize=6)
    fig.colorbar(im, label="log(1 + count)")
    ax.set_title("Co-occurrence of findings")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    pd.DataFrame(m, index=DISEASES, columns=DISEASES).to_csv(path.replace(".png", ".csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="путь к .zip или к распакованной папке")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--min-mean", type=float, default=20.0, help="темнее -> отбраковка")
    ap.add_argument("--max-mean", type=float, default=235.0, help="светлее -> отбраковка")
    ap.add_argument("--limit", type=int, default=0, help="для быстрого теста: первые N снимков")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("[1/5] Индексирую файлы...")
    index, csv_bytes, zip_path = build_file_index(a.src)
    df = build_metadata(csv_bytes)
    if a.limit:
        df = df.head(a.limit).copy()
    missing = [x for x in df["image"] if x not in index]
    if missing:
        print(f"[WARN] нет файла для {len(missing)} строк CSV, они будут удалены")
        df = df[df["image"].isin(index)].reset_index(drop=True)
    df = df.reset_index(drop=True)
    print(f"      Картинок в таблице: {len(df):,}")

    print(f"[2/5] Уменьшаю до {a.size}x{a.size} ({a.workers} процессов)...")
    out_npy = os.path.join(a.out, f"images_{a.size}.npy")
    arr = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.uint8,
                                    shape=(len(df), a.size, a.size))
    jobs = [(i, index[n], a.size) for i, n in enumerate(df["image"])]
    means = np.zeros(len(df), np.float32)
    decoded_ok = np.ones(len(df), bool)
    with Pool(a.workers, initializer=_init_worker, initargs=(zip_path,)) as pool:
        for i, small, mean, ok in tqdm(pool.imap_unordered(_load_one, jobs, chunksize=64),
                                       total=len(jobs)):
            arr[i] = small; means[i] = mean; decoded_ok[i] = ok
    arr.flush(); del arr

    print("[3/5] Контроль качества...")
    df["mean_brightness"] = means
    too_dark = (means < a.min_mean) & decoded_ok
    too_bright = (means > a.max_mean) & decoded_ok
    df["quality_ok"] = decoded_ok & ~too_dark & ~too_bright
    print(f"      Слишком тёмных: {int(too_dark.sum())}, слишком светлых: {int(too_bright.sum())}, "
          f"битых: {int((~decoded_ok).sum())}")

    print("[4/5] Разбиение по пациентам...")
    df = make_splits(df)
    ok, leak_txt = leakage_check(df)
    print(leak_txt)
    open(os.path.join(a.out, "leakage_check.txt"), "w", encoding="utf-8").write(leak_txt)

    print("[5/5] Статистика и сохранение...")
    prev = prevalence_table(df)
    print(prev.to_string(index=False))
    prev.to_csv(os.path.join(a.out, "prevalence.csv"), index=False)
    cooccurrence_plot(df, os.path.join(a.out, "cooccurrence.png"))
    df.to_csv(os.path.join(a.out, "metadata.csv"), index=False)
    report = {
        "n_images": len(df), "n_patients": int(df.patient_id.nunique()),
        "n_train_images": int((df.split == "train").sum()),
        "n_test_images": int((df.split == "test").sum()),
        "filtered_too_dark": int(too_dark.sum()), "filtered_too_bright": int(too_bright.sum()),
        "filtered_corrupt": int((~decoded_ok).sum()),
        "brightness_thresholds": [a.min_mean, a.max_mean],
        "no_leakage": bool(ok), "seed": SEED, "size": a.size,
    }
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=2)
    print("Готово ->", a.out)


if __name__ == "__main__":
    main()
