FROM python:3.11

WORKDIR /workspace


COPY requirements.txt .

ARG TORCH_INDEX_URL=https://pypi.org/simple

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends git make \
    && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["python", "--version"]
