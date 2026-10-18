# Image Preprocessing Pipeline

What happens to an NIH ChestX-ray14 image between the PNG file and the model, why each step is there, and what we checked.

The code that does this is the training code: `prep_data.py` (offline step), `dataset.py` (online step), and `config.py` (all values). Every value below matches those files as of 18 October 2026. If a value changes there, update this page and `scripts/make_preprocessing_figures.py`.

## TL;DR

- **Offline, once:** decode the PNG as grayscale → brightness quality check → resize to 224×224 (`INTER_AREA`) → store as `uint8` in `images_224.npy`.
- **Online, per sample:** [training only: random rotation/scale/shift/flip] → CLAHE (clip limit 2.0, 8×8 tiles) → ÷255 → copy to 3 channels → ImageNet mean/std. The model gets a `float32` tensor of shape 3×224×224.
- Validation, test, and API inference use the same steps without augmentation, so the same image always gives the same tensor.
- **Quality check:** 0 of 112,120 images were removed (thresholds 20 and 235). The thresholds only catch blank or broken files.
- **Order matters:** CLAHE runs *after* resizing. Running it before resizing changes the input by about 5 grey levels per pixel, so inference must use the same order.
- The before/after CLAHE comparison is visual only. We have not measured whether CLAHE improves AUROC.

## Pipeline

**Figure:** `01_pipeline_overview.png` (diagram) and `02_pipeline_steps.png` (one image through every step with its histogram).

| # | Step | Setting | Code | Why |
| --- | --- | --- | --- | --- |
| 1 | Decode | `cv2.IMREAD_GRAYSCALE`; NIH PNGs are 1024×1024, 8-bit | `prep_data.py` | X-rays have one channel. 71 of 4,999 sample files (1.4%) are stored as RGBA; reading as grayscale handles them. |
| 2 | Quality check | Mean brightness μ of the raw image; keep 20 ≤ μ ≤ 235. Files that fail to decode are flagged too. | `prep_data.py` → `metadata.csv` columns `mean_brightness`, `quality_ok` | Drop blank, badly exposed, or broken files. Measured on the raw image, before any contrast change. |
| 3 | Resize | 224×224, `INTER_AREA` | `prep_data.py` | Input size of ImageNet-pretrained DenseNet-121. `INTER_AREA` averages pixel blocks, which gives the least aliasing when shrinking about 4.6×. The images are square, so nothing is stretched. |
| 4 | Store | `uint8` array `images_224.npy` (~5.6 GB) + `metadata.csv` | `prep_data.py` | Decoding 42 GB of PNGs every epoch is too slow. Training reads the array through a memory map. |
| 5 | Augmentation | Rotation ±10°, scale 0.9–1.1, shift ±5% (±11 px), horizontal flip with p = 0.5; bilinear, mirrored border | `dataset.py`, `CXRDataset._augment` | **Training only.** Imitates differences in patient position and reduces overfitting. |
| 6 | CLAHE | `clipLimit=2.0`, `tileGridSize=(8, 8)`, so each tile is 28×28 px | `dataset.py`, `apply_clahe` | Raises local contrast in dense regions (mediastinum, behind the heart, below the diaphragm) without blowing out the lungs. See below. |
| 7 | Scale and channels | ÷255 → [0, 1]; the gray image is copied into R, G, and B | `dataset.py` | Pretrained weights expect 3 channels. |
| 8 | Normalize | mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) | `dataset.py` | Match the statistics the ImageNet weights were trained with. |

Output: `float32`, shape 3×224×224. Values can range from −2.12 (black) to 2.64 (white).

Steps 1–4 run once. Steps 5–8 run every time a sample is loaded, so CLAHE and augmentation settings can change without rebuilding `images_224.npy`.

## CLAHE

CLAHE (Contrast Limited Adaptive Histogram Equalization) splits the image into tiles and equalizes the histogram of each tile separately. The *clip limit* caps how much contrast a tile may gain, so noise in flat areas is not amplified too much. Neighbouring tiles are blended with bilinear interpolation, so no tile edges appear.

### Settings

**Figure:** `06_clahe_parameters.png`.

- **Clip limit 2.0, 8×8 tiles** are the values used in the OpenCV CLAHE tutorial, a common starting point.
- Clip limits of 4 and 8 make lung texture look harsh and amplify noise in soft tissue.
- 16×16 tiles on a 224×224 image are only 14 px wide, and the result looks blotchy. 2×2 tiles behave almost like a global adjustment.
- Global histogram equalization (`cv2.equalizeHist`) brightens the whole image and washes out the lungs, so we did not use it.

These settings were **not tuned** on validation results. We picked them before training.

### Before and after

**Figures:** `03_clahe_before_after.png` (six findings), `04_clahe_nodule_examples.png` (Nodule), `05_clahe_lesion_zoom.png` (Mass with NIH bounding boxes, zoomed), `07_intensity_histograms.png` (all sample images).

What we see:

- **Lungs:** vessels, airway walls, and rib edges look sharper.
- **Dense regions:** the mediastinum, the area behind the heart, and the area below the diaphragm show more structure, because these regions get their own tile histograms instead of being pushed to white.
- **Mass with bounding box:** in all three boxed Mass cases in our sample, the edge between the lesion and the surrounding lung looks clearer to us after CLAHE.
- **Nodule:** NIH does not give a location for most Nodule images (none in our sample has a bounding box), so we cannot say whether a nodule became more visible. Size is also a problem. At 224×224 one pixel is about 1.9 mm (median field width 420 mm ÷ 224), so a 1 cm nodule covers only about 5 pixels. This may be part of why Nodule is one of the weakest classes (validation AUROC 0.747), but we have not tested this.
- **Side effects:** noise and film grain in flat areas are amplified, burned-in text and markers (`L`, `PORTABLE`) become brighter, and black borders become dark grey (≈3) instead of 0.

Numbers over the 4,999 sample images, at 224×224:

| Measure | Before CLAHE | After CLAHE |
| --- | ---: | ---: |
| Median of mean brightness | 126.2 | 137.9 |
| Median global contrast (std of pixel values) | 60.0 | 63.0 |
| Images with higher contrast after CLAHE | – | 83.1% |

> ⚠️ These are visual observations and image statistics. They do **not** show that CLAHE helps diagnosis or the model. To measure that, train the same fold with `use_clahe=False` and compare AUROC.

## Quality check

**Figure:** `10_brightness_quality_check.png`.

- **Rule:** mean brightness of the raw 1024×1024 image. Below 20 = too dark, above 235 = too bright. Files that cannot be decoded also fail. Failed images stay in `metadata.csv` with `quality_ok = False`. `get_split(..., only_quality_ok=True)` leaves them out of training, validation, and test.
- **Full dataset** (Damir's `prep_data.py` run, reported in the model card): 112,120 images, **0 removed by the brightness rule**. The number of broken files is in that run's `report.json` (`filtered_corrupt`).
- **Sample** (`images_001`, 4,999 images): mean brightness ranges from 58 to 216; 98% of images are between 88 and 180. No image comes close to either threshold.
- The rule and its thresholds were fixed before comparing models. Do not adjust them based on test results.

What the check does **not** catch:

- `00001085_000.png` is labelled `PA` but is a **lateral** view.
- Unusually framed images pass too: the darkest image in our sample (μ = 58) is a small, tilted film inside a wide black border, and another dark one is a zoomed-in supine portable film.
- Catching these needs a different check (view or body-part classifier, or manual review). This is out of scope for the midterm.

**Two brightness groups.** The brightness histogram has two peaks, around 110 and around 160. PA and AP images appear in both. PA images are slightly brighter (median 129.6 vs. 121.5 for AP). The peaks probably come from different scanners or post-processing; we did not investigate. CLAHE and normalization do not fully remove these global differences, so the model could use overall brightness as a shortcut. Grad-CAM and subgroup analysis should look out for this.

## Training augmentation

**Figure:** `09_training_augmentation.png`.

- Augmentation runs before CLAHE, so CLAHE always sees the final geometry. Each epoch gets a new random version of every training image.
- **Mirrored border:** filling the border with a mirror image (`BORDER_REFLECT_101`) creates reflected copies of cables and edges near the image border.
- **Horizontal flip:** flipping moves the heart to the right and mirrors the `L` marker. The result is anatomically unusual. This may confuse findings that depend on left and right, such as Cardiomegaly. A run without flipping would show whether it helps.

## Training, validation, and inference give the same input

**Figure:** `08_resize_clahe_order.png`.

| Check | Result |
| --- | --- |
| Same image processed twice (validation path) | Identical tensors |
| Output shape, type, values | 3×224×224, `float32`, all finite, range [−2.10, 2.64] for the example image |
| API decoding (`PIL` `convert("L")`) vs. training decoding (`cv2.IMREAD_GRAYSCALE`) | Identical pixels for all 4,999 sample images, including the 71 RGBA files |
| Resize → CLAHE (training) vs. CLAHE → resize | Per-image mean difference: median 5.4 grey levels, 95th percentile 7.1. **Not interchangeable.** |

So the model `config.json` and the API must use exactly:

```json
"preprocessing": {
  "grayscale": true,
  "resize": [224, 224],
  "resize_strategy": "direct_resize",
  "interpolation": "INTER_AREA",
  "clahe": true,
  "clahe_settings": {"clip_limit": 2.0, "tile_grid_size": [8, 8], "apply_before_resize": false},
  "channels": "repeat_grayscale_3_times",
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225]
}
```

Not checked yet: JPEG uploads (lossy compression gives slightly different pixels than the original PNG) and DICOM input.

## Reproducing the figures

The figures were made from `images_001.tar.gz` (4,999 images, 1,335 patients) from the [NIH download page](https://nihcc.app.box.com/v/ChestXray-NIHCC), together with `Data_Entry_2017.csv` and `BBox_List_2017.csv`. The script also accepts the full Kaggle zip.

```bash
python scripts/make_preprocessing_figures.py --images <folder with PNGs or Kaggle zip> --metadata <Data_Entry_2017.csv> --bbox <BBox_List_2017.csv>
```

It needs `opencv-python-headless`, `numpy`, `pandas`, `matplotlib`, and `Pillow` from `requirements.txt` and takes about 3 minutes for 5,000 images. Example images are picked with seed 42.

Output goes to `outputs/preprocessing/`, which Git ignores. **Do not commit the figures:** they contain NIH images.

| File | Contents |
| --- | --- |
| `01_pipeline_overview.png` | Pipeline diagram |
| `02_pipeline_steps.png` | One image after each step, with histograms |
| `03_clahe_before_after.png` | Before/after CLAHE for six findings |
| `04_clahe_nodule_examples.png` | Before/after CLAHE for four Nodule images |
| `05_clahe_lesion_zoom.png` | Mass bounding boxes, full image and zoom, before/after |
| `06_clahe_parameters.png` | Clip limit and tile size comparison |
| `07_intensity_histograms.png` | Average histogram and per-image contrast, before/after |
| `08_resize_clahe_order.png` | Resize → CLAHE vs. CLAHE → resize |
| `09_training_augmentation.png` | Random training augmentations |
| `10_brightness_quality_check.png` | Brightness distribution with thresholds; darkest/brightest images |
| `summary.json` | All numbers on this page |
| `examples_used.csv` | Which images appear in which figure |
| `per_image_stats.csv` | Brightness, contrast, and checks for every image |

## Limitations and open questions

1. **No CLAHE ablation.** We don't know whether CLAHE improves AUROC. Suggested run: fold 0, gamma 1, `use_clahe=False`.
2. **CLAHE settings were not tuned.**
3. **The quality check is a safety net only.** It does not detect wrong view, wrong body part, or rotation.
4. **224×224 is coarse for small findings.** A 1 cm nodule is about 5 pixels.
5. **The figures come from one archive.** `images_001` has 1,335 patients and no Nodule bounding boxes. The full-dataset counts come from the `prep_data.py` report.
6. **Horizontal flip and mirrored borders** create anatomically unusual training images; their effect is not measured.
7. **DICOM input** (with `pydicom` and PHI handling) is not implemented.
