# VisionX_CXR-CAD
Multi-label chest X-ray disease detection using NIH ChestX-ray14, DenseNet, EfficientNet, Grad-CAM, FastAPI, and Streamlit.

> **Educational project, not for diagnosis.** VisionX is a student capstone prototype. Its scores are based on NIH labels and must not be used for clinical decisions.

**Try it:** [VisionX dashboard](TODO-DASHBOARD-URL) · [API status page](TODO-API-URL)

## Results

Trained on NIH ChestX-ray14 (112,115 images from 30,805 patients after quality filtering). Patients are split 85% / 15% into train and a fixed test set, with 5-fold patient-wise cross-validation inside train; no patient appears in more than one set. Focal Loss with gamma 0 (plain weighted BCE) was chosen by validation AUROC.

**Fixed test set** (16,724 images, 4,621 patients; 14 diseases, mean over diseases):

| Model | Mean AUROC | Mean AUPRC |
| --- | --- | --- |
| DenseNet-121, single fold | 0.830 ± 0.004 | 0.249 |
| DenseNet-121, 5-fold average | 0.845 | 0.270 |
| EfficientNet-B0, 5-fold average | 0.841 | 0.269 |
| **Ensemble (DenseNet-121 + EfficientNet-B0, 10 models) with flip TTA** | **0.848** | **0.277** |

| Check | Target | Result |
| --- | --- | --- |
| Mean AUROC | ≥ 0.80 | 0.848 |
| Calibration (mean ECE) | ≤ 0.10 | 0.085 raw → 0.043 after per-class temperature scaling |
| Subgroup gap (sex, age, view) | < 0.10 | largest gap 0.019 (age 60+) |
| 5-fold validation AUROC | | DenseNet-121 0.832 ± 0.003 · EfficientNet-B0 0.827 ± 0.003 |
| External check (CheXpert valid, 202 images, 7 shared diseases) | | 0.854 vs. 0.859 on NIH for the same diseases |

Best and weakest diseases on the test set: Hernia 0.939, Emphysema 0.936, Cardiomegaly 0.921 · Infiltration 0.703, Nodule 0.767, Pneumonia 0.771. In 81 of 100 checked cases the Grad-CAM peak falls inside the lungs. The deployed API returns calibrated scores; detection thresholds are not applied yet.

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

## Run with Docker

The API image downloads the trained ensemble from the team's private Hugging Face repository during the build, so you need a Hugging Face read token with access to it (ask the team leader) and the release values from [the deployment guide](docs/deployment.md#2-configure-the-render-api-service) (`HF_REPO_ID`, `HF_REVISION`, `MODEL_ARCHIVE_NAME`, `MODEL_ARCHIVE_SHA256`, `MODEL_BUNDLE_DIR`). Never commit the token.

1. Start Docker Desktop. From the repository root, save the token in a file that Git ignores and set the release values:

   ```bash
   printf '%s' 'hf_your_token' > hf_token
   export HF_REPO_ID=... HF_REVISION=... MODEL_ARCHIVE_NAME=... MODEL_ARCHIVE_SHA256=... MODEL_BUNDLE_DIR=/workspace/models/...
   ```

2. Build both images:

   ```bash
   docker build --secret id=hf_token,src=hf_token \
     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
     --build-arg HF_REPO_ID --build-arg HF_REVISION --build-arg MODEL_ARCHIVE_NAME \
     --build-arg MODEL_ARCHIVE_SHA256 --build-arg MODEL_BUNDLE_DIR \
     -t visionx-api .
   docker build -f Dockerfile.streamlit \
     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
     -t visionx-dashboard .
   ```

3. Start the API, then the dashboard, on a shared network:

   ```bash
   docker network create visionx
   docker run -d --name visionx-api --network visionx -p 127.0.0.1:8000:8000 \
     -e MODEL_BUNDLE_DIR visionx-api \
     python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   docker run -d --name visionx-dashboard --network visionx -p 127.0.0.1:8501:8501 \
     -e API_URL=http://visionx-api:8000 visionx-dashboard \
     python -m streamlit run src/dashboard/app.py --server.address=0.0.0.0 --server.port=8501
   ```

4. Open http://localhost:8501 for the dashboard and http://localhost:8000/health for the API (it may take about a minute to load the 10 models). Upload a PNG, JPEG or DICOM chest X-ray. DICOM files have patient identifiers removed before inference.

5. Stop everything with `docker rm -f visionx-dashboard visionx-api && docker network rm visionx`.

To run without Docker, put an extracted bundle (a folder with `inference_config.json` and `models/`) under `models/`, set `MODEL_BUNDLE_DIR` in `.env` (see `.env.example`), and run `make api` and `make dashboard` in two terminals.

## Development commands

Run these from the repository root inside the Dev Container:

| Command | Purpose |
| --- | --- |
| `make help` | Show the command help |
| `make install` | Install dependencies from `requirements.txt` |
| `make lint` | Check Python lint rules |
| `make lint-fix` | Automatically fix supported lint issues |
| `make format` | Format Python files |
| `make format-check` | Check formatting without changing files |
| `make typecheck` | Check types in `src/` |
| `make test` | Run tests in `tests/` |
| `make check` | Run lint, formatting verification, type checking, and tests |
| `make dashboard` | Start the Streamlit dashboard locally on port `8501` |
| `make api` | Start FastAPI locally on port `8000` with auto-reload |
| `make docker-build` | Build Docker Compose service images |
| `make docker-up` | Build images and start Docker Compose services in the background |
| `make docker-down` | Stop and remove Docker Compose containers and networks |
| `make docker-logs` | Follow logs from Docker Compose services |

`make docker-build` and `make docker-up` use `compose.yaml`, which does not pass the Hugging Face token the API image now needs, so they currently fail; use the steps in [Run with Docker](#run-with-docker) instead.

Run these commands from the project root. Docker commands require Docker to be running (for example, through Docker Desktop on macOS). Press `Ctrl+C` to stop following logs; the services will keep running.

Ruff and mypy use the settings in `pyproject.toml`.
`make check` stops on the first failure and does not modify files.

Dependencies are installed during the container build, so `make install`
is usually only needed after changing `requirements.txt`.

## Contributing

Before starting a task, read the [GitHub Workflow Guides](docs/github-workflow/README.md) for commit style, task creation, branch naming, pull requests, reviews, and conflict resolution.

## Team

We are a team of Computer Science and Engineering students at Sogang University working on this project as part of the 2026-2 Capstone Design course.

**Track:** AI Healthcare  
**Project:** 흉부 X-ray 다중 질환 탐지 시스템 (CXR-CAD) 개발  
**English Title:** Chest X-ray Multi-Disease Detection System (CXR-CAD)

### Members
| Member | GitHub | Contribution |
| --- | --- | --- |
| Ha Anna Maria (Team Leader) | [ha-anna](https://github.com/ha-anna) | TODO-CONFIRM: project setup and workflow, Dev Container and Docker, Streamlit dashboard, dashboard–API–model integration, ensemble inference, disease-specific Grad-CAM, Render and Hugging Face deployment |
| Maksutova Aibike | [bimoonity](https://github.com/bimoonity) | TODO-CONFIRM: NIH ChestX-ray14 dataset research, preprocessing pipeline documentation |
| Zaripov Damir | [dkapro](https://github.com/dkapro) | TODO-CONFIRM: project dependencies, baseline data preparation and DenseNet-121 / Focal Loss training and evaluation scripts |
| Mukhiddinova Malika | [likamuradovna](https://github.com/likamuradovna) | TODO-CONFIRM: FastAPI `/health` and `/predict` endpoints, requirements versions |
| Sadullaeva Madina | [madina1727](https://github.com/madina1727) | TODO-CONFIRM: repository structure, pytest test suite, DICOM upload with de-identification, dashboard safety notices, README |


## Dev Container setup

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and make sure it is running.
2. Install [VS Code](https://code.visualstudio.com/) and the **Dev Containers** extension.
3. Clone the repository and open the folder in VS Code.
4. Press `F1` and select **Dev Containers: Reopen in Container**.
5. Wait for the first build to finish. Python and all dependencies are then available in the container.


## Deployment

See [Deploy VisionX on Render](docs/deployment.md) for ensemble packaging,
Hugging Face access, and Render setup.