# Deploy VisionX on Render with Hugging Face

GitHub hosts the application code. A private Hugging Face model repository hosts the ensemble bundle. The Render API Docker build downloads and verifies the archive, then FastAPI loads the extracted bundle at startup. Streamlit sends prediction requests to FastAPI.

## 1. Package and upload an ensemble

From the VisionX repository root, replace `YYYY_MM_DD` with the training date used in the bundle directory name. The directory should contain `inference_config.json` and a `models/` subdirectory with the ten checkpoint files.

```bash
bundle_dir=YYYY_MM_DD_ensemble
archive="/tmp/visionx-${bundle_dir}.tar.gz"

tar -czf "$archive" -C models "$bundle_dir"
tar -tzf "$archive"
shasum -a 256 "$archive"
```

Check that the versioned directory is the top-level archive entry and that the config and all checkpoints are present. Upload the exact archive you hashed to the private Hugging Face model repository under **Files and versions**. Record the archive filename, the upload commit SHA, and the archive SHA-256. Recompute the checksum if you repack the bundle.

## 2. Configure the Render API service

Set these values on the FastAPI service. The example names below use the packaging convention in step 1. Use the actual filename and directory from your bundle.

| Setting | Value |
| --- | --- |
| `HF_REPO_ID` | Private Hugging Face repository ID, such as `<org>/<model-repo>` |
| `HF_REVISION` | Full Hugging Face commit SHA for the uploaded archive |
| `MODEL_ARCHIVE_NAME` | Exact uploaded filename, such as `visionx-YYYY_MM_DD_ensemble.tar.gz` |
| `MODEL_ARCHIVE_SHA256` | SHA-256 of the exact uploaded archive |
| `MODEL_BUNDLE_DIR` | Extracted directory, such as `/workspace/models/YYYY_MM_DD_ensemble` |

The revision pins the repository state. The checksum verifies the archive bytes. Update the release settings together when deploying or rolling back a bundle.

Create a fine-grained Hugging Face token with read access to the model repository. Under **Environment → Secret Files**, add a file named `hf_token` containing only the token. Keep the token, local archives, and checkpoints out of Git.

- [Hugging Face access tokens](https://huggingface.co/docs/hub/security-tokens)
- [Render Docker build secrets](https://render.com/docs/docker-secrets)

## 3. Download the bundle during the Docker build

In the Dockerfile configured for the FastAPI service, add this to the final API image stage after installing `curl` and copying the application source. Render makes the service environment variables available as Docker build arguments. The token is read from a separate build secret.

```dockerfile
ARG HF_REPO_ID
ARG HF_REVISION
ARG MODEL_ARCHIVE_NAME
ARG MODEL_ARCHIVE_SHA256
ARG MODEL_BUNDLE_DIR

RUN --mount=type=secret,id=hf_token,required=true \
    curl --fail --silent --show-error --location --retry 3 \
      --header "Authorization: Bearer $(cat /run/secrets/hf_token)" \
      --output /tmp/visionx-ensemble.tar.gz \
      "https://huggingface.co/${HF_REPO_ID}/resolve/${HF_REVISION}/${MODEL_ARCHIVE_NAME}" \
    && printf '%s  %s\n' "$MODEL_ARCHIVE_SHA256" /tmp/visionx-ensemble.tar.gz | sha256sum -c - \
    && mkdir -p /workspace/models \
    && tar -xzf /tmp/visionx-ensemble.tar.gz -C /workspace/models \
    && test -f "${MODEL_BUNDLE_DIR}/inference_config.json" \
    && rm /tmp/visionx-ensemble.tar.gz
```

Keep the API image private because it contains the extracted weights. Use `.dockerignore` to exclude local checkpoint and archive copies from the build context. [Docker on Render](https://render.com/docs/docker)

## 4. Verify the deployment

Set the Render health check path to `/health` and rebuild the API. Check the build log for successful checksum verification and the runtime log for `Application startup complete` after the bundle loads.

From the repo root, set `api` to the URL shown in the Render dashboard and run a prediction using the demo image.

```bash
api=https://YOUR-API-SERVICE.onrender.com
curl -fsS --max-time 60 "$api/health"
curl -fsS --max-time 120 -F 'file=@data/test.png' \
  "$api/predict" -o /tmp/visionx-response.json
python3 -c 'import json; r = json.load(open("/tmp/visionx-response.json")); print("Bundle", r["model_version"]); print("Scores", len(r["scores"])); print("Heatmap present", bool(r.get("heatmap_base64")))'
```

Check for HTTP 200, the expected model version, 14 scores, and a heatmap when Grad-CAM is enabled. Verify the result in Streamlit too.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Archive download returns 401, 403, or 404 | Check token scope, secret filename, repository ID, revision, and archive filename. |
| SHA-256 mismatch | Confirm the configured checksum is for the exact archive at the pinned revision. |
| Bundle missing at startup | Check the archive's top-level directory and `MODEL_BUNDLE_DIR`. |
| Root URL returns 404 | Use `/health` to check the API. |
| Health succeeds but prediction returns 502 | Check Render Events at the request timestamp. The ensemble loaded on a 512 MB instance, but a prediction exceeded the memory limit. Test with Grad-CAM temporarily disabled to isolate its memory cost, or verify a full prediction on a larger API instance. |
