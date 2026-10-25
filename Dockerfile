# syntax=docker/dockerfile:1.4
# ==============================================================================
# OmniVision Agent (Zero-Shot Visual Agent v3) - Production Multi-Stage Container
# ==============================================================================

# Stage 1: Build Dependencies
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --user -e ".[dev]"

# Stage 2: Runtime Production Image
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    OMNIVISION_DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Security Hardening: Non-root application user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser

WORKDIR /app

# Copy installed wheels and user site-packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application source code
COPY --chown=appuser:appuser . /app

# Create cache and artifact directories
RUN mkdir -p /app/.cache /app/eval_artifacts /app/reports && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 7860

# Production Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

ENTRYPOINT ["omnivision"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
