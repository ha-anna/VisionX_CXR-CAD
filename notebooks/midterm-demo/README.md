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
1. python prep_data.py
2. python train.py
3. python evaluate.py

## Setup
- Dev subset / fold: <>
- Data: not included, see data/README.md
- Checkpoint: not committed, <>
- Dependencies: see requirements.txt in repo root