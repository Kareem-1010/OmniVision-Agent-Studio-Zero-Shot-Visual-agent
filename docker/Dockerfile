# OmniVision Agent CUDA 12.2 Production Container
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMNIVISION_DEVICE=cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy dependency specifications
COPY pyproject.toml .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e ".[dev]"

# Copy source repository
COPY . .

EXPOSE 8000 7860

CMD ["python", "scripts/run_phase1_demo.py"]
