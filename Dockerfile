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

ARG HF_REVISION=main

RUN --mount=type=secret,id=hf_token,required=true \
    curl --fail --silent --show-error --location --retry 3 \
      --header "Authorization: Bearer $(cat /run/secrets/hf_token)" \
      --output /tmp/visionx-ensemble.tar.gz \
      "https://huggingface.co/visionx-cxr-cad/visionx-ensemble/resolve/${HF_REVISION}/visionx-ensemble-2026-11-23.tar.gz" \
    && echo "50704ce4336c111abfc8368a1bdb1ef09605518dd637a5b15a1581e01ad4b6cf  /tmp/visionx-ensemble.tar.gz" | sha256sum -c - \
    && mkdir -p /workspace/models \
    && tar -xzf /tmp/visionx-ensemble.tar.gz -C /workspace/models \
    && test -f /workspace/models/2026_11_23_ensemble/inference_config.json \
    && rm /tmp/visionx-ensemble.tar.gz

CMD ["python", "--version"]
