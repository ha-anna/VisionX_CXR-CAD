# NIH ChestX-ray14: Dataset Summary

What the team needs to know about the dataset before we write preprocessing and training code.

All numbers below were computed from `Data_Entry_2017.csv`, `train_val_list.txt`, `test_list.txt`, and `BBox_List_2017.csv`. We have not opened the image archives yet, so image-level facts come from the dataset documentation.

## TL;DR

- **112,120** frontal chest X-rays from **30,805** patients. Each image is a 1024×1024 grayscale PNG.
- One metadata CSV holds every label and every demographic field. Labels sit in a single text column separated by `|`.
- The task is **multi-label**: 14 diseases plus `No Finding`. 40% of abnormal images have more than one disease.
- **Patients have many images** (up to 184). We must **split by `Patient ID`, never by image**.
- The official train/val vs. test split is already patient-wise, but **there is no official validation split**. We have to make our own from `train_val_list.txt`, also patient-wise.
- The labels were **text-mined from radiology reports** by NLP. They are not radiologist-annotated, so expect label noise.

## Files

| File | What it is | Do we need it? |
| --- | --- | --- |
| `images_001` … `images_012` | 12 folders/archives of PNG images (~45 GB total). In the Kaggle layout they look like `images_00X/images/*.png`. | Yes |
| `Data_Entry_2017.csv` | One row per image: labels, patient ID, age, sex, view, original size | **Yes (main file)** |
| `train_val_list.txt` | 86,524 image names in the official train+val pool | Yes |
| `test_list.txt` | 25,596 image names in the official test set | Yes |
| `BBox_List_2017.csv` | 984 bounding boxes on 880 images, 8 diseases | Optional, for checking Grad-CAM |
| `README_ChestXray.pdf`, `FAQ_CHESTXRAY.pdf`, `LOG_CHESTXRAY.pdf` | Official documentation and change log | Read once |

## `Data_Entry_2017.csv` columns

| Column | Example | Meaning / notes |
| --- | --- | --- |
| `Image Index` | `00000001_001.png` | File name. It always equals `<Patient ID padded to 8 digits>_<Follow-up # padded to 3>.png`. |
| `Finding Labels` | `Cardiomegaly\|Emphysema` | Labels separated by `\|`, or exactly `No Finding` |
| `Follow-up #` | `1` | Visit index for that patient, starting at 0 (max 183) |
| `Patient ID` | `1` | **Key for splitting.** 30,805 unique values |
| `Patient Age` | `58` | Age in years. Has errors; see Limitations. |
| `Patient Gender` | `M` / `F` | Consistent across all images of the same patient |
| `View Position` | `PA` / `AP` | Projection. AP is usually a bedside or portable film. |
| `OriginalImage[Width`, `Height]` | `2682`, `2749` | Size before NIH resized to 1024×1024 |
| `OriginalImagePixelSpacing[x`, `y]` | `0.143`, `0.143` | mm per pixel in the original image |

> ⚠️ **CSV header quirk:** the commas inside `OriginalImage[Width,Height]` split the header into broken names (`OriginalImage[Width`, `Height]`, …). A trailing comma also creates an empty `Unnamed: 11` column. Rename the columns right after loading, and drop the last column.

## Labels

Each image has either `No Finding` or one or more of the 14 diseases. `No Finding` never appears together with a disease. The table below counts images: a multi-label image is counted once for each of its labels.

| Label | Images | % of all | Train/val | Test | Patients |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Finding | 60,361 | 53.8% | – | – | – |
| Infiltration | 19,894 | 17.7% | 13,782 | 6,112 | 8,035 |
| Effusion | 13,317 | 11.9% | 8,659 | 4,658 | 4,275 |
| Atelectasis | 11,559 | 10.3% | 8,280 | 3,279 | 4,981 |
| Nodule | 6,331 | 5.6% | 4,708 | 1,623 | 3,394 |
| Mass | 5,782 | 5.2% | 4,034 | 1,748 | 2,568 |
| Pneumothorax | 5,302 | 4.7% | 2,637 | 2,665 | 1,487 |
| Consolidation | 4,667 | 4.2% | 2,852 | 1,815 | 2,150 |
| Pleural_Thickening | 3,385 | 3.0% | 2,242 | 1,143 | 2,006 |
| Cardiomegaly | 2,776 | 2.5% | 1,707 | 1,069 | 1,566 |
| Emphysema | 2,516 | 2.2% | 1,423 | 1,093 | 1,046 |
| Edema | 2,303 | 2.1% | 1,378 | 925 | 1,073 |
| Fibrosis | 1,686 | 1.5% | 1,251 | 435 | 1,260 |
| Pneumonia | 1,431 | 1.3% | 876 | 555 | 1,008 |
| Hernia | 227 | 0.2% | 141 | 86 | 134 |

- **Multi-label:** 20,796 images (18.5% of all, 40.2% of abnormal ones) carry 2–9 labels. There are 836 distinct label combinations.
- **Model target:** a 14-dim 0/1 vector. We should **not** treat `No Finding` as a 15th class: it is simply "all 14 = 0".
- **Spelling:** use the exact strings `Pleural_Thickening` (with an underscore), and note that `BBox_List_2017.csv` writes `Infiltrate`, not `Infiltration`.
- **Class imbalance is severe.** Hernia has only 134 patients, so its metrics will be unstable. We will likely need class weights or a weighted loss.

## Patients and multiple images

- 17,503 patients (57%) have one image. 13,302 have two or more: mean 3.6, median 1, max 184.
- Patients with ≥10 images contribute **48% of all images**.
- 9,464 patients have different labels at different follow-ups, because disease changes over time.
- **Consequence:** an image-level random split would put the same person in both train and test and inflate our AUC. Every split we make (train/val/test, and k-fold if we use it) must group by `Patient ID`.

## Official split

| | Images | Patients | No Finding | AP view | Mean Follow-up # |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train_val_list.txt` | 86,524 | 28,008 | 58.4% | 35.0% | 5.1 |
| `test_list.txt` | 25,596 | 2,797 | 38.5% | 56.6% | 20.3 |

- No image overlap and **no patient overlap**, and the two lists together cover all 112,120 images.
- The test set is **not** a random sample. It has fewer patients, sicker patients, more AP films, and many follow-ups per patient. Our test scores will not match our validation scores, and that is expected.
- All 880 bounding-box images are in the test set.

## Demographics (for subgroup analysis)

| Field | Distribution |
| --- | --- |
| Sex (images) | M 63,340 (56.5%) · F 48,780 (43.5%) |
| Sex (patients) | M 16,630 · F 14,175 |
| View | PA 67,310 (60%) · AP 44,810 (40%) |
| Age (images) | median 49, IQR 35–59. Bins: <18: 5,241 · 18–39: 30,694 · 40–59: 49,137 · 60–79: 25,913 · ≥80: 1,119 |

Subgroups we can evaluate: **sex**, **age group**, and **view position (AP/PA)**. View position is strongly tied to the labels. AP images have fewer normal images than PA (47.0% vs. 58.4% `No Finding`) and far more Edema (4.5% vs. 0.4%). The model could learn "AP film = sick" as a shortcut, so we should report results per view.

The metadata has **no race/ethnicity, no hospital/scanner, no dates, and no clinical history**.

## Limitations and open questions for discussion

1. **Noisy labels.** NIH extracted the labels from reports with NLP and expects their accuracy to be above 90%. Test-set AUCs are measured against these noisy labels too.
2. **Invalid ages.** 16 images have ages 148–155 or 411–414. *Proposal:* set them to missing and exclude them from age-subgroup analysis.
3. **Two CSV versions are in circulation.** We found two copies of `Data_Entry_2017.csv` that differ:
   - **181 images have different labels** between the two copies.
   - One copy stores ages as text (`058Y`, `011M`, `001D`). The other stores plain numbers, where 27 infant ages in months or days became "years" (for example, `011M` → `11`).
   - NIH has also published a `Data_Entry_2017_v2020.csv`.
   - **Decision needed:** everyone should use the same file, e.g. the one from the Kaggle download we train with. We should record its checksum in the repo.
4. **No official validation split.** *Proposal:* split `train_val_list.txt` patient-wise, about 85/15 or 90/10, fix the random seed, and commit the resulting list of patient IDs or image names.
5. **Keep or drop follow-ups?** One patient has up to 184 images. Options: keep all, or cap/weight images per patient so a few patients don't dominate training.
6. **Image channels.** The PNGs are documented as grayscale. We still need to check after download whether any files load with extra channels, and always convert to one channel (or repeat to 3 for ImageNet backbones).
7. **Bounding boxes cover only 8 of the 14 diseases**, and all 880 boxed images are in the test set. They are suitable only for a qualitative Grad-CAM check.

## Sources

- Wang et al., *ChestX-ray8: Hospital-scale Chest X-ray Database…*, CVPR 2017
- [NIH Chest X-ray dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [Hugging Face mirror (dataset card, v2020 CSV)](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset)
- Metadata copies used for the statistics: [TRKuan/cxr8](https://github.com/TRKuan/cxr8) (numeric ages, plus split lists) and [gregwchase/nih-chest-xray](https://github.com/gregwchase/nih-chest-xray) (text ages)
