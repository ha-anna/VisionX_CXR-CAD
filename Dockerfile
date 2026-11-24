FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.txt .

ARG TORCH_INDEX_URL=https://pypi.org/simple

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        --index-url "${TORCH_INDEX_URL}" \
        torch==2.14.1 torchvision==0.29.1 \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY . .

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
    && test -f "$MODEL_BUNDLE_DIR/inference_config.json" \
    && rm /tmp/visionx-ensemble.tar.gz

CMD ["python", "--version"]
