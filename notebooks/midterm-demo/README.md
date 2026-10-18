# Midterm demo: baseline pipeline (issue #42)

Scripts (not notebooks) for the NIH ChestX-ray14 baseline.

## Files
- prep_data.py — manifest, patient-level splits
- dataset.py — preprocessing (CLAHE, 224x224, normalization), data loaders
- losses.py — Focal Loss
- metrics.py — AUROC and other metrics
- train.py — DenseNet-121 training
- evaluate.py — validation evaluation
- config.py — configuration

## How to run
python prep_data.py
python train.py --fold 0 --gamma 2 --name d121_g2_f0
python evaluate.py

## Setup
- Dev subset / fold: full dataset (112,120 images), fold 0 of 5
- Data: not included, see data/README.md
- Checkpoint: not committed, <checkpoint available from dkapro on request>
- Dependencies: see requirements.txt in repo root
