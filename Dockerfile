FROM python:3.11

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

CMD ["python", "--version"]
