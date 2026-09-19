# syntax=docker/dockerfile:1
#
# Self-contained ANN2SNN dashboard image.
#
#   docker build -t ann2snn-dashboard .
#   docker run --rm -p 8080:8080 ann2snn-dashboard
#   # then open http://localhost:8080
#
# Stage 1 builds the Vite frontend; stage 2 is a CPU-only Python runtime with the
# simulation engine, the Flask API, ffmpeg (for MP4 export) and the built static
# bundle. No OpenClaw/DinD base image is required.

# --------------------------------------------------------------------------- #
# Stage 1 - build the frontend
# --------------------------------------------------------------------------- #
FROM node:20-bookworm-slim AS web

WORKDIR /web
COPY web/package.json web/package-lock.json web/.npmrc ./
# --cache overrides the workspace-pinned cache path in .npmrc (not present here).
RUN npm ci --no-audit --no-fund --cache /root/.npm
COPY web/ ./
# vite.config.js emits to ../server/static, i.e. /server/static from WORKDIR /web.
RUN npm run build

# --------------------------------------------------------------------------- #
# Stage 2 - CPU runtime: engine + Flask API + ffmpeg + built dashboard
# --------------------------------------------------------------------------- #
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    ANN2SNN_THREADS=2 \
    ANN2SNN_WEIGHTS=/app/data/weights.pt \
    ANN2SNN_WORKERS=2 \
    PORT=8080

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first: PyPI's linux `torch` wheel drags in ~2.5 GB of CUDA
# libraries, which this workload never uses. Once installed it satisfies the
# engine's `torch>=2.0` requirement and the later `pip install` leaves it alone.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url "${TORCH_INDEX_URL}" torch

COPY pyproject.toml README.md ./
COPY sim_engine ./sim_engine
# `dev` adds pytest so the suite can be run inside the shipped image.
RUN pip install ".[dashboard,video,dev]"

# Pre-distil the default brains so the learned controllers balance out of the box
# (and the first request is instant). If this ever fails the image still builds
# and the server auto-distils on first use instead.
RUN mkdir -p /app/data \
 && python3 -c "from sim_engine.config import EngineConfig; from sim_engine.engine import Engine; Engine(EngineConfig(device='cpu'), train=True).save_weights('/app/data/weights.pt')" \
    || echo "warmup distillation skipped; the server will distil on first use"

COPY server ./server
COPY tools ./tools
COPY tests ./tests
COPY docs ./docs
COPY examples ./examples

# Built frontend from stage 1.
COPY --from=web /server/static ./server/static

RUN mkdir -p /app/videos /app/data \
 && useradd --create-home --shell /usr/sbin/nologin app \
 && chown -R app:app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# gunicorn's sync workers are fine here; /api/export/mp4 blocks one worker for a
# few seconds, so keep a generous timeout.
CMD ["sh", "-c", "exec gunicorn -b 0.0.0.0:${PORT} --workers ${ANN2SNN_WORKERS} --timeout 300 server.wsgi:application"]
