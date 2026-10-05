# VisionX_CXR-CAD
Multi-label chest X-ray disease detection using NIH ChestX-ray14, DenseNet, EfficientNet, Grad-CAM, FastAPI, and Streamlit.

## Project Structure

```text
src/
  data/         dataset loading, patient-wise splits, preprocessing
  models/       DenseNet / EfficientNet architectures and training
  evaluation/   metrics and threshold selection
  xai/          Grad-CAM and other explainability methods
  api/          FastAPI inference service
  dashboard/    Streamlit dashboard
tests/          automated tests (mirror the src/ layout)
configs/        experiment and training configuration
notebooks/      exploration notebooks (clear outputs before committing)
scripts/        command-line entry points (download, train, evaluate)
docs/           project and workflow documentation
data/           local datasets — ignored by Git
models/         local weights and checkpoints — ignored by Git
```

Datasets, model weights (`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`), and generated outputs (`outputs/`, `predictions/`, `gradcam_outputs/`, `runs/`, `wandb/`) are excluded by `.gitignore`. Never commit patient data.

## Contributing

Before starting a task, read the [GitHub Workflow Guides](docs/github-workflow/README.md) for commit style, task creation, branch naming, pull requests, reviews, and conflict resolution.

## Team

We are a team of Computer Science and Engineering students at Sogang University working on this project as part of the 2026-2 Capstone Design course.

**Track:** AI Healthcare  
**Project:** 흉부 X-ray 다중 질환 탐지 시스템 (CXR-CAD) 개발  
**English Title:** Chest X-ray Multi-Disease Detection System (CXR-CAD)

### Members
- Ha Anna Maria (Team Leader) - [ha-anna](https://github.com/ha-ana)
- Maksutova Aibike - [bimoonity](https://github.com/bimoonity)
- Zaripov Damir - [dkapro](https://github.com/dkapro)
- Mukhiddinova Malika - [likamuradovna](https://github.com/likamuradovna)
- Sadullaeva Madina - [madina1727](https://github.com/madina1727)
