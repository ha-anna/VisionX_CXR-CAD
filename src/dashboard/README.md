# VisionX Streamlit UI

Standalone starter matching the proposed two-column screen. No checkpoint is loaded in the dashboard. The API is a separate service and is not included here.

## Organized layout

- `dashboard.py`: page orchestration, upload handling, analysis action, and session state.
- `api_client.py`: HTTP requests and response validation.
- `image_utils.py`: upload/Base64 decoding.
- `components.py`: image viewer, score bars, and result panels.
- `styles.py`: centralized styling.
- `settings.py`: labels, API URL, and upload limit.
- `preview.py`: explicitly illustrative responses.
- `app.py`: compatibility launcher for the original run command.

The Overview layout preserves the original two-column screen. Image focus uses a wider page and puts the results below the image viewer. In Image focus, select Compare both, Original only, or Grad-CAM only. A single image uses the full viewer width. Switching layouts retains the current API response and does not rerun inference. Images keep their aspect ratio; this is a larger viewer, not a pan/zoom tool and cannot recover detail absent from the supplied image.

## Run

From this directory, using your project Python environment:

```
make dashboard
```

For integration into the existing repository, merge only missing dependencies into its requirements file and preserve the team's agreed versions. Streamlit configuration is resolved from the working directory: copy `.streamlit/config.toml` to the repository root if launching from there.

The default API URL is `http://localhost:8000`. Set `API_URL` to change it. Local Streamlit can reach a Docker API with port 8000 published. If Streamlit also runs in Docker, use the API's container/service hostname instead of localhost.

Use the preview toggle to inspect the design without an API. Its scores are explicitly illustrative; uploads do not generate simulated predictions. Analyze is disabled in preview mode.

## API agreement

`POST /predict` must accept a multipart field named `file` and return:

```json
{
  "predictions": {
    "Atelectasis": 0.1,
    "Cardiomegaly": 0.1,
    "Effusion": 0.1,
    "Infiltration": 0.1,
    "Mass": 0.1,
    "Nodule": 0.1,
    "Pneumonia": 0.1,
    "Pneumothorax": 0.1,
    "Consolidation": 0.1,
    "Edema": 0.1,
    "Emphysema": 0.1,
    "Fibrosis": 0.1,
    "Pleural_Thickening": 0.1,
    "Hernia": 0.1
  },
  "detected_diseases": [],
  "top1_disease": "Atelectasis",
  "gradcam_heatmap": "data:image/png;base64,...",
  "inference_time_ms": 123,
  "model_version": "baseline-v1"
}
```

These are illustrative schema values, not model results. `gradcam_heatmap` accepts raw Base64 or a data URL. By default it is interpreted as an already blended overlay. If the API returns a colored heatmap alone, set `GRADCAM_IS_OVERLAY=false`; the app resizes and blends it onto the original image. Confirm map orientation/alignment with the model team. Grayscale activation matrices need to be converted to a colored PNG by the API.

Detections are supplied by the API, not calculated from chart color bands. No findings does not mean a clinically normal image. Results are cleared when the upload or preview mode changes. The backend must independently validate uploads.

## Validation status

Python syntax and TOML configuration were checked. Streamlit/requests are not installed in the authoring runtime, so live rendering and real API integration remain unverified. Start in preview mode, then check real PNG/JPEG requests, Grad-CAM alignment, invalid uploads, API timeouts, and changing the uploaded image. The CSS intentionally keeps native uploader/button behavior; its internal Streamlit selectors may require adjustment after upgrades.
