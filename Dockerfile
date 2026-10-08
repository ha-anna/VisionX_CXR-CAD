FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

ARG TORCH_INDEX_URL=https://pypi.org/simple

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        --index-url "${TORCH_INDEX_URL}" \
        torch==2.14.1 torchvision==0.29.1 \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "--version"]
