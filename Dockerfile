FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /opt/raku
COPY requirements.txt ./
RUN python -m pip install --upgrade "pip==23.3" \
    && python -m pip install -r requirements.txt

COPY . .
RUN python -m pip install --no-deps -e .

ENTRYPOINT ["raku"]
CMD ["--help"]
