"""Generate the figures and numbers that document the image preprocessing pipeline.

The steps and values mirror the training code (prep_data.py -> images_224.npy,
dataset.py -> CXRDataset, config.py):

    decode PNG as grayscale -> brightness check on the raw image -> resize to 224x224
    (INTER_AREA) -> [training only: random affine + horizontal flip] -> CLAHE
    (clip 2.0, 8x8 tiles) -> divide by 255 -> repeat to 3 channels -> ImageNet mean/std

If a value changes in config.py, change it here too, then re-run this script.
Figures show NIH images, so they go to outputs/ (ignored by Git). Do not commit them.

Example:
    python scripts/make_preprocessing_figures.py --images D:/cxr_sample/extracted/images \
        --metadata D:/cxr_sample/Data_Entry_2017.csv --bbox D:/cxr_sample/BBox_List_2017.csv
"""

import argparse
import io
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

# ---------- pipeline settings (keep in sync with config.py / prep_data.py / dataset.py) ----------
IMG_SIZE = 224
INTERPOLATION = cv2.INTER_AREA
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), np.float32)[:, None, None]
IMAGENET_STD = np.array((0.229, 0.224, 0.225), np.float32)[:, None, None]
MIN_MEAN, MAX_MEAN = 20.0, 235.0  # prep_data.py --min-mean / --max-mean
AUG_ROTATION = 10.0  # degrees, uniform in [-10, 10]
AUG_SCALE = (0.9, 1.1)
AUG_SHIFT = 0.05  # fraction of the image width
AUG_FLIP_P = 0.5

# Labelled "PA" in the NIH metadata but actually a lateral view; kept out of the example figures.
# The brightness check does not catch it (see docs/dataset/preprocessing.md).
NOT_FRONTAL = {"00001085_000.png"}

REPO_ROOT = Path(__file__).resolve().parents[1]
USED_COLOR = "#d62728"


# ---------- reading ----------
class ImageSource:
    """PNG files from a folder (searched recursively) or straight from the Kaggle zip."""

    def __init__(self, path):
        self.zip = None
        if str(path).lower().endswith(".zip"):
            self.zip = zipfile.ZipFile(path)
            names = [n for n in self.zip.namelist() if n.lower().endswith(".png")]
            self.index = {os.path.basename(n): n for n in names}
        else:
            self.index = {p.name: p for p in Path(path).rglob("*.png")}

    def read(self, name):
        ref = self.index[name]
        return self.zip.read(ref) if self.zip else Path(ref).read_bytes()


def decode_gray(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("decode failed")
    return img


def load_metadata(path, available):
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Image Index": "image",
            "Finding Labels": "labels",
            "Patient ID": "patient_id",
            "View Position": "view",
        }
    )[["image", "labels", "patient_id", "view"]]
    df = df[df["image"].isin(available)].reset_index(drop=True)
    df["label_list"] = df["labels"].str.split("|")
    return df


# ---------- pipeline steps ----------
def resize(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=INTERPOLATION)


def apply_clahe(img, clip=CLAHE_CLIP, tile=CLAHE_TILE):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(img)


def to_model_input(img_u8):
    x = img_u8.astype(np.float32) / 255.0
    x = np.repeat(x[None], 3, axis=0)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def augment(img, rng):
    """Same random affine + flip as CXRDataset._augment (training only)."""
    h, w = img.shape
    angle = rng.uniform(-AUG_ROTATION, AUG_ROTATION)
    scale = rng.uniform(*AUG_SCALE)
    tx, ty = rng.uniform(-AUG_SHIFT, AUG_SHIFT, 2) * w
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    flip = rng.random() < AUG_FLIP_P
    if flip:
        img = img[:, ::-1]
    return np.ascontiguousarray(img), dict(angle=angle, scale=scale, tx=tx, ty=ty, flip=flip)


def analyse(source, name):
    """One pass over a raw image: QC value, 224 version and the checks reported in summary.json."""
    try:
        data = source.read(name)
        raw = decode_gray(data)
        small = resize(raw)
        after = apply_clahe(small)
        clahe_first = resize(apply_clahe(raw))  # the other possible order, for comparison
        pil = Image.open(io.BytesIO(data))
        pil_gray = np.asarray(pil.convert("L"))  # how the API decodes uploads
        return dict(
            image=name,
            ok=True,
            height=raw.shape[0],
            width=raw.shape[1],
            mean_raw=float(raw.mean()),
            png_mode=pil.mode,
            pil_equals_cv2=bool(np.array_equal(pil_gray, raw)),
            mean_before=float(small.mean()),
            mean_after=float(after.mean()),
            std_before=float(small.std()),
            std_after=float(after.std()),
            order_mad=float(np.abs(after.astype(np.int16) - clahe_first).mean()),
            hist_before=np.bincount(small.ravel(), minlength=256) / small.size,
            hist_after=np.bincount(after.ravel(), minlength=256) / after.size,
            small=small,
        )
    except Exception as e:  # a broken file must not stop the whole run
        return dict(image=name, ok=False, error=str(e))


# ---------- plotting helpers ----------
def show(ax, img, title=None, ylabel=None, highlight=False, **kw):
    handle = ax.imshow(
        img,
        cmap="gray",
        vmin=kw.pop("vmin", 0),
        vmax=kw.pop("vmax", 255),
        interpolation=kw.pop("interpolation", "antialiased"),
        **kw,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(
            title,
            fontsize=10,
            color=USED_COLOR if highlight else "black",
            fontweight="bold" if highlight else "normal",
        )
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(highlight)
        s.set_color(USED_COLOR)
        s.set_linewidth(2.5)
    return handle


def save(fig, out, name):
    fig.savefig(out / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")


def pick(df, label, n, rng, exclude=()):
    """n images of one label (single-label and PA first, so the finding is less mixed)."""
    has = df[df["label_list"].apply(lambda ls: label in ls) & ~df["image"].isin(exclude)]
    has = has.assign(single=has["label_list"].str.len() == 1, pa=has["view"] == "PA")
    has = has.assign(r=rng.random(len(has))).sort_values(
        ["single", "pa", "r"], ascending=[False, False, True]
    )
    return has.head(n)


def short_labels(labels, limit=28):
    return labels if len(labels) <= limit else labels[: limit - 1] + "…"


# ---------- figures ----------
def fig_overview(out):
    fig, ax = plt.subplots(figsize=(15, 6.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    def box(x, y, w, h, title, body, face, dashed=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=face,
                edgecolor="#333333",
                linewidth=1.3,
                linestyle="--" if dashed else "-",
            )
        )
        ax.text(
            x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=11, fontweight="bold"
        )
        ax.text(x + w / 2, y + h / 2 - 0.25, body, ha="center", va="center", fontsize=9.5)

    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4),
        )

    ax.text(
        0.4,
        6.35,
        "OFFLINE  ·  prep_data.py  ·  run once for the whole dataset",
        fontsize=11,
        fontweight="bold",
        color="#1f4e79",
        va="center",
    )
    offline = [
        ("1. Read PNG", "1024×1024, 8-bit\ndecoded as grayscale\n(straight from the zip)"),
        (
            "2. Quality check",
            f"mean brightness μ of\nthe raw image\nkeep {MIN_MEAN:.0f} ≤ μ ≤ "
            f"{MAX_MEAN:.0f}\nbroken files flagged",
        ),
        ("3. Resize", f"{IMG_SIZE}×{IMG_SIZE}\nINTER_AREA\n(area averaging)"),
        ("4. Save", "uint8 array\nimages_224.npy\n+ metadata.csv"),
    ]
    w, gap, y = 2.85, 0.65, 3.55
    for i, (t, b) in enumerate(offline):
        x = 0.4 + i * (w + gap)
        box(x, y, w, 2.45, t, b, "#dbe9f6")
        if i:
            arrow(x - gap, y + 1.22, x, y + 1.22)

    ax.text(
        2.05,
        3.05,
        "ONLINE  ·  dataset.py  ·  every time a sample is loaded",
        fontsize=11,
        fontweight="bold",
        color="#2d6a2d",
        va="center",
    )
    online = [
        (
            "5. Augmentation",
            f"rotation ±{AUG_ROTATION:.0f}°, scale "
            f"{AUG_SCALE[0]}–{AUG_SCALE[1]}\nshift ±{AUG_SHIFT:.0%}, "
            f"h-flip p={AUG_FLIP_P}\nTRAINING ONLY",
            "#fde9d2",
            True,
        ),
        (
            "6. CLAHE",
            f"clipLimit {CLAHE_CLIP}\ntiles {CLAHE_TILE[0]}×{CLAHE_TILE[1]}\n"
            f"(on the {IMG_SIZE}×{IMG_SIZE} image)",
            "#dff0d8",
            False,
        ),
        ("7. Scale + 3 ch", "÷ 255 → [0, 1]\ngray copied into\nR, G, B channels", "#dff0d8", False),
        (
            "8. Normalize",
            "ImageNet mean/std\nmean (0.485, 0.456, 0.406)\nstd (0.229, 0.224, 0.225)",
            "#dff0d8",
            False,
        ),
        (
            "9. Model input",
            f"float32 tensor\n3×{IMG_SIZE}×{IMG_SIZE}\n→ DenseNet-121",
            "#e7e7e7",
            False,
        ),
    ]
    w, gap, y = 2.5, 0.43, 0.25
    for i, (t, b, face, dashed) in enumerate(online):
        x = 0.4 + i * (w + gap)
        box(x, y, w, 2.45, t, b, face, dashed)
        if i:
            arrow(x - gap, y + 1.22, x, y + 1.22)

    # images_224.npy feeds the dataset
    x_npy = 0.4 + 3 * (2.85 + 0.65) + 2.85 / 2
    ax.plot([x_npy, x_npy, 1.65], [3.55, 3.3, 3.3], color="#333333", lw=1.4)
    arrow(1.65, 3.3, 1.65, 2.72)
    ax.text(
        14.6,
        2.95,
        "validation, test and API inference skip step 5;\n"
        "everything else is identical and deterministic",
        ha="right",
        va="center",
        fontsize=9.5,
        style="italic",
        color="#444444",
    )
    save(fig, out, "01_pipeline_overview.png")


def fig_steps(out, source, row):
    raw = decode_gray(source.read(row.image))
    small = resize(raw)
    after = apply_clahe(small)
    x = to_model_input(after)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), gridspec_kw={"height_ratios": [3, 1.2]})
    show(
        axes[0, 0],
        raw,
        f"1. Original PNG  {raw.shape[1]}×{raw.shape[0]}\n"
        f"mean brightness μ = {raw.mean():.0f}  →  QC pass",
    )
    show(axes[0, 1], small, f"2. Resize → {IMG_SIZE}×{IMG_SIZE}\nINTER_AREA")
    show(axes[0, 2], after, f"3. CLAHE\nclip {CLAHE_CLIP}, {CLAHE_TILE[0]}×{CLAHE_TILE[1]} tiles")
    im = show(
        axes[0, 3],
        x[0],
        "4. Model input (channel R of 3)\n÷255, ImageNet mean/std",
        vmin=float(x[0].min()),
        vmax=float(x[0].max()),
    )
    fig.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.03)
    # the black border makes a tall spike at the darkest values; cut it so the rest is visible
    for ax, img in zip(axes[1, :3], (raw, small, after)):
        h = np.bincount(img.ravel(), minlength=256) / img.size
        ax.bar(np.arange(256), h, width=1, color="#4a4a4a")
        ax.set_ylim(0, h[8:].max() * 1.15)
        ax.set_xlim(0, 255)
        ax.set_yticks([])
        ax.set_xlabel("pixel value (0–255)")
    h, _, _ = axes[1, 3].hist(x[0].ravel(), bins=100, color="#4a4a4a")
    axes[1, 3].set_ylim(0, h[3:].max() * 1.15)
    axes[1, 3].set_yticks([])
    axes[1, 3].set_xlabel(f"normalized value  [{x.min():.2f}, {x.max():.2f}]")
    fig.suptitle(
        f"One image through the pipeline  ({row.image}, {row.labels}, {row.view})", fontsize=13
    )
    fig.tight_layout()
    save(fig, out, "02_pipeline_steps.png")


def fig_before_after(out, rows, small, name, title):
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 6.6))
    axes = np.asarray(axes).reshape(2, n)
    for j, r in enumerate(rows.itertuples()):
        img = small[r.image]
        show(
            axes[0, j],
            img,
            f"{short_labels(r.labels)}\n{r.view} · {r.image}",
            ylabel="Before CLAHE" if j == 0 else None,
        )
        show(axes[1, j], apply_clahe(img), ylabel="After CLAHE" if j == 0 else None)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    save(fig, out, name)


def fig_lesion_zoom(out, boxes, small):
    k = IMG_SIZE / 1024.0
    fig, axes = plt.subplots(len(boxes), 4, figsize=(13, 3.5 * len(boxes)))
    axes = np.asarray(axes).reshape(len(boxes), 4)
    for i, b in enumerate(boxes.itertuples()):
        before = small[b.image]
        after = apply_clahe(before)
        x, y, w, h = b.x * k, b.y * k, b.w * k, b.h * k
        side = max(w, h) * 2.2
        cx, cy = x + w / 2, y + h / 2
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(IMG_SIZE, cx + side / 2)), int(min(IMG_SIZE, cy + side / 2))
        for j, (img, tag) in enumerate(((before, "Before CLAHE"), (after, "After CLAHE"))):
            show(
                axes[i, j],
                img,
                tag if i == 0 else None,
                ylabel=f"{b.label}\n{b.image}" if j == 0 else None,
            )
            axes[i, j].add_patch(Rectangle((x, y), w, h, fill=False, ec="#ffcc00", lw=1.6))
            show(
                axes[i, j + 2],
                img[y0:y1, x0:x1],
                f"{tag} — zoom on the box" if i == 0 else None,
                interpolation="nearest",
            )
            axes[i, j + 2].add_patch(
                Rectangle((x - x0, y - y0), w, h, fill=False, ec="#ffcc00", lw=1.6)
            )
    fig.suptitle(
        f"Lesions with NIH bounding boxes, at the {IMG_SIZE}×{IMG_SIZE} resolution the "
        "model sees (zoom = enlarged pixels)",
        fontsize=12,
    )
    fig.tight_layout()
    save(fig, out, "05_clahe_lesion_zoom.png")


def fig_parameters(out, img):
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.4))
    show(axes[0, 0], img, "No CLAHE")
    for ax, clip in zip(axes[0, 1:], (1.0, 2.0, 4.0, 8.0)):
        used = clip == CLAHE_CLIP
        show(
            ax,
            apply_clahe(img, clip=clip),
            f"clip {clip}, tiles 8×8" + ("  (used)" * used),
            highlight=used,
        )
    show(axes[1, 0], cv2.equalizeHist(img), "Global histogram\nequalization (not used)")
    for ax, t in zip(axes[1, 1:], (2, 4, 8, 16)):
        used = (t, t) == CLAHE_TILE
        show(
            ax,
            apply_clahe(img, tile=(t, t)),
            f"clip 2.0, tiles {t}×{t}" + ("  (used)" * used),
            highlight=used,
        )
    axes[0, 0].set_ylabel("changing clipLimit", fontsize=11)
    axes[1, 0].set_ylabel("changing tile grid", fontsize=11)
    fig.suptitle(
        "CLAHE settings: stronger clip / smaller tiles = more local contrast, "
        "but also more amplified noise",
        fontsize=12,
    )
    fig.tight_layout()
    save(fig, out, "06_clahe_parameters.png")


def fig_histograms(out, stats):
    hb = np.mean(np.stack(stats["hist_before"]), axis=0)
    ha = np.mean(np.stack(stats["hist_after"]), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    axes[0].plot(hb, label="before CLAHE", color="#1f77b4")
    axes[0].plot(ha, label="after CLAHE", color=USED_COLOR)
    axes[0].set_xlim(0, 255)
    axes[0].set_xlabel("pixel value (0–255)")
    axes[0].set_ylabel("share of pixels")
    axes[0].set_title(f"Average intensity histogram ({len(stats)} images, {IMG_SIZE}×{IMG_SIZE})")
    axes[0].legend()
    bins = np.linspace(0, max(stats.std_before.max(), stats.std_after.max()) + 2, 60)
    axes[1].hist(stats.std_before, bins=bins, alpha=0.6, label="before CLAHE", color="#1f77b4")
    axes[1].hist(stats.std_after, bins=bins, alpha=0.6, label="after CLAHE", color=USED_COLOR)
    axes[1].set_xlabel("global contrast = std of pixel values")
    axes[1].set_ylabel("images")
    axes[1].set_title(
        f"Per-image contrast: median {stats.std_before.median():.1f} → "
        f"{stats.std_after.median():.1f}"
    )
    axes[1].legend()
    fig.tight_layout()
    save(fig, out, "07_intensity_histograms.png")


def fig_order(out, source, row, stats):
    raw = decode_gray(source.read(row.image))
    used = apply_clahe(resize(raw))
    other = resize(apply_clahe(raw))
    diff = np.abs(used.astype(np.int16) - other)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    show(axes[0], used, "Resize → CLAHE  (used in training)", highlight=True)
    show(axes[1], other, "CLAHE on 1024×1024 → resize")
    im = axes[2].imshow(diff, cmap="magma", vmin=0, vmax=max(1, int(np.percentile(diff, 99))))
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title(f"|difference|, mean {diff.mean():.1f} grey levels", fontsize=10)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    axes[3].hist(stats.order_mad, bins=50, color="#4a4a4a")
    axes[3].set_xlabel("mean |difference| per image (grey levels)")
    axes[3].set_ylabel("images")
    axes[3].set_title(
        f"All {len(stats)} images: median {stats.order_mad.median():.1f}", fontsize=10
    )
    fig.suptitle(
        "The order of resize and CLAHE matters: inference must use the training order", fontsize=12
    )
    fig.tight_layout()
    save(fig, out, "08_resize_clahe_order.png")


def fig_augmentation(out, img, seed):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.6))
    axes = axes.ravel()
    show(axes[0], apply_clahe(img), "Validation / test / API\n(no augmentation)", highlight=True)
    for ax in axes[1:]:
        aug, p = augment(img, rng)
        show(
            ax,
            apply_clahe(aug),
            f"rot {p['angle']:+.1f}°, scale {p['scale']:.2f}\n"
            f"shift ({p['tx']:+.0f}, {p['ty']:+.0f}) px" + (", flipped" if p["flip"] else ""),
        )
    fig.suptitle(
        "Training-only augmentation (a new random version every epoch), then CLAHE", fontsize=12
    )
    fig.tight_layout()
    save(fig, out, "09_training_augmentation.png")


def fig_quality(out, stats, small):
    fig = plt.figure(figsize=(16, 6.4))
    gs = fig.add_gridspec(2, 5, width_ratios=[2.6, 0.15, 1, 1, 1], hspace=0.3)
    ax = fig.add_subplot(gs[:, 0])
    bins = np.arange(0, 256, 2)
    views = ["PA", "AP"]
    ax.hist(
        [stats.loc[stats.view == v, "mean_raw"] for v in views],
        bins=bins,
        stacked=True,
        color=["#4a4a4a", "#9e9e9e"],
        label=[f"{v} view" for v in views],
    )
    ax.legend(loc="center right")
    ax.axvspan(0, MIN_MEAN, color=USED_COLOR, alpha=0.18)
    ax.axvspan(MAX_MEAN, 255, color=USED_COLOR, alpha=0.18)
    for t in (MIN_MEAN, MAX_MEAN):
        ax.axvline(t, color=USED_COLOR, ls="--")
    dark = int((stats.mean_raw < MIN_MEAN).sum())
    bright = int((stats.mean_raw > MAX_MEAN).sum())
    ax.text(
        MIN_MEAN + 3,
        ax.get_ylim()[1] * 0.92,
        f"too dark\n(< {MIN_MEAN:.0f}): {dark}",
        color=USED_COLOR,
        fontsize=10,
    )
    ax.text(
        MAX_MEAN - 3,
        ax.get_ylim()[1] * 0.92,
        f"too bright\n(> {MAX_MEAN:.0f}): {bright}",
        color=USED_COLOR,
        fontsize=10,
        ha="right",
    )
    ax.set_xlim(0, 255)
    ax.set_xlabel("mean brightness of the raw 1024×1024 image")
    ax.set_ylabel("images")
    ax.set_title(f"Brightness quality check on {len(stats)} images")
    order = stats.sort_values("mean_raw")
    for k, (r, tag) in enumerate(
        [(r, "darkest") for r in order.head(3).itertuples()]
        + [(r, "brightest") for r in order.tail(3).itertuples()]
    ):
        a = fig.add_subplot(gs[k // 3, 2 + k % 3])
        show(a, small[r.image], f"{tag}: μ = {r.mean_raw:.0f}")
    fig.tight_layout()
    save(fig, out, "10_brightness_quality_check.png")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", required=True, help="folder with NIH PNGs, or the Kaggle zip")
    ap.add_argument("--metadata", required=True, help="Data_Entry_2017.csv")
    ap.add_argument("--bbox", help="BBox_List_2017.csv (optional, for the lesion zoom figure)")
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "preprocessing"))
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    source = ImageSource(a.images)
    df = load_metadata(a.metadata, set(source.index))
    if a.limit:
        df = df.head(a.limit)
    print(f"[1/3] {len(df):,} images with metadata ({df.patient_id.nunique():,} patients)")

    with ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(lambda n: analyse(source, n), df["image"]))
    failed = [r for r in results if not r["ok"]]
    stats = pd.DataFrame([r for r in results if r["ok"]])
    small = dict(zip(stats["image"], stats.pop("small")))
    stats["view"] = stats["image"].map(df.set_index("image")["view"])
    ok_ids = set(stats.loc[(stats.mean_raw >= MIN_MEAN) & (stats.mean_raw <= MAX_MEAN), "image"])
    good = df[df["image"].isin(ok_ids) & ~df["image"].isin(NOT_FRONTAL)]
    print(
        f"[2/3] analysed; {len(failed)} broken, {len(df) - len(failed) - len(ok_ids)} rejected "
        "by the brightness check"
    )

    print("[3/3] figures ->", out)
    used = []

    def record(figure, rows):
        used.extend(
            dict(figure=figure, image=r.image, labels=r.labels, view=r.view)
            for r in rows.itertuples()
        )

    fig_overview(out)
    step_row = pick(good, "Nodule", 1, rng)
    fig_steps(out, source, next(step_row.itertuples()))
    record("02_pipeline_steps", step_row)

    findings = ["Cardiomegaly", "Effusion", "Pneumothorax", "Mass", "Emphysema", "No Finding"]
    grid = pd.concat([pick(good, d, 1, rng) for d in findings])
    fig_before_after(
        out,
        grid,
        small,
        "03_clahe_before_after.png",
        f"CLAHE before / after on {IMG_SIZE}×{IMG_SIZE} inputs (what the model sees)",
    )
    record("03_clahe_before_after", grid)

    nodules = pick(good, "Nodule", 4, rng, exclude=step_row["image"])
    fig_before_after(
        out,
        nodules,
        small,
        "04_clahe_nodule_examples.png",
        "Nodule examples, before / after CLAHE (NIH labels, no location given)",
    )
    record("04_clahe_nodule_examples", nodules)

    if a.bbox:
        bb = pd.read_csv(a.bbox).iloc[:, :6]
        bb.columns = ["image", "label", "x", "y", "w", "h"]
        bb = bb[bb["image"].isin(ok_ids)]
        focal = bb[bb["label"].isin(["Nodule", "Mass"])]
        boxes = (focal if len(focal) else bb).head(4)
        if len(boxes):
            fig_lesion_zoom(out, boxes, small)
            record("05_clahe_lesion_zoom", df.set_index("image").loc[boxes["image"]].reset_index())
        else:
            print("  no bounding boxes for these images, skipped 05_clahe_lesion_zoom.png")

    example = small[step_row.image.iloc[0]]
    fig_parameters(out, example)
    fig_histograms(out, stats)
    fig_order(out, source, next(step_row.itertuples()), stats)
    aug_row = pick(good, "No Finding", 1, rng)
    fig_augmentation(out, small[aug_row.image.iloc[0]], a.seed)
    record("09_training_augmentation", aug_row)
    fig_quality(out, stats, small)

    # numbers for the documentation
    x = to_model_input(apply_clahe(example))
    x_again = to_model_input(apply_clahe(resize(decode_gray(source.read(step_row.image.iloc[0])))))
    dark = stats.loc[stats.mean_raw < MIN_MEAN, "image"].tolist()
    bright = stats.loc[stats.mean_raw > MAX_MEAN, "image"].tolist()
    q = stats.mean_raw.quantile([0, 0.01, 0.5, 0.99, 1]).round(1)
    summary = {
        "pipeline": {
            "decode": "cv2.IMREAD_GRAYSCALE",
            "resize": [IMG_SIZE, IMG_SIZE],
            "interpolation": "INTER_AREA",
            "clahe_clip_limit": CLAHE_CLIP,
            "clahe_tile_grid": list(CLAHE_TILE),
            "clahe_after_resize": True,
            "mean": [round(v, 3) for v in IMAGENET_MEAN.ravel().tolist()],
            "std": [round(v, 3) for v in IMAGENET_STD.ravel().tolist()],
            "brightness_thresholds": [MIN_MEAN, MAX_MEAN],
        },
        "sample": {
            "images": len(df),
            "patients": int(df.patient_id.nunique()),
            "views": df["view"].value_counts().to_dict(),
        },
        "raw_sizes": (stats.width.astype(str) + "x" + stats.height.astype(str))
        .value_counts()
        .to_dict(),
        "png_modes": stats.png_mode.value_counts().to_dict(),
        "broken_files": [dict(image=r["image"], error=r["error"]) for r in failed],
        "quality_check": {
            "too_dark": len(dark),
            "too_bright": len(bright),
            "rejected_ids": dark + bright,
            "mean_brightness_quantiles": {f"q{int(k * 100)}": v for k, v in q.items()},
        },
        "clahe_effect_224": {
            "median_mean_before": round(stats.mean_before.median(), 1),
            "median_mean_after": round(stats.mean_after.median(), 1),
            "median_std_before": round(stats.std_before.median(), 1),
            "median_std_after": round(stats.std_after.median(), 1),
            "share_images_with_higher_std": round(
                float((stats.std_after > stats.std_before).mean()), 3
            ),
        },
        "resize_clahe_order_mean_abs_diff": {
            "median": round(stats.order_mad.median(), 2),
            "p95": round(stats.order_mad.quantile(0.95), 2),
        },
        "api_decoder_parity": {
            "pil_convert_L_equals_cv2_grayscale": int(stats.pil_equals_cv2.sum()),
            "checked": len(stats),
        },
        "model_input_check": {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "finite": bool(np.isfinite(x).all()),
            "min": round(float(x.min()), 3),
            "max": round(float(x.max()), 3),
            "deterministic": bool(np.array_equal(x, x_again)),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(used).to_csv(out / "examples_used.csv", index=False)
    stats.drop(columns=["hist_before", "hist_after"]).to_csv(
        out / "per_image_stats.csv", index=False
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "quality_check",
                    "clahe_effect_224",
                    "resize_clahe_order_mean_abs_diff",
                    "api_decoder_parity",
                    "model_input_check",
                )
            },
            indent=2,
        )[:3000]
    )


if __name__ == "__main__":
    main()
