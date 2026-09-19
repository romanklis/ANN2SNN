# ANN2SNN — dashboard, backend and demo-video workflows.
#
# Quick start (bring the whole solution up):
#
#   make up            # build the frontend, then serve the dashboard on :8080
#   make up-docker     # same, but via the self-contained image (docker compose)
#
# Run `make up` inside the Dev/DinD image (torch CPU + Flask + Node 20 + ffmpeg).
# Everything else (tests, videos, docker) is listed by `make help`.

PY ?= python3
PORT ?= 8080
WORKERS ?= 2
TIMEOUT ?= 300
IMAGE ?= ann2snn-dashboard:latest
HOST ?= 0.0.0.0

# `make` with no target brings the solution up.
.DEFAULT_GOAL := up

.PHONY: help up up-docker check-env install test test-engine test-server test-render \
        web-install web-build serve serve-dev \
        videos videos-train videos-fast \
        docker-build docker-run docker-shell docker-videos docker-stop \
        compose-up compose-down clean

help:
	@echo "make up            build the frontend and serve the dashboard on :$(PORT)"
	@echo "make up-docker     build+run the container (auto free port; no compose)"
	@echo "make install       install engine + dashboard/video/dev extras"
	@echo "make test          run all test suites (engine, server, renderer)"
	@echo "make web-install   npm ci in web/ (skipped if already installed)"
	@echo "make web-build     build the frontend into server/static/"
	@echo "make serve         gunicorn backend + built dashboard on :$(PORT)"
	@echo "make serve-dev     Flask dev server (debug) on :$(PORT)"
	@echo "make videos-train  distil and render all demo MP4s into videos/"
	@echo "make videos-fast   render untrained baselines (quick)"
	@echo "make docker-build  build the self-contained image ($(IMAGE))"
	@echo "make docker-run    build+run it, auto-picking a free port from $(PORT)"
	@echo "make docker-videos render the demo MP4s inside the image"
	@echo "make docker-shell  open a shell in the image"
	@echo "make compose-up    optional compose workflow (needs a free \$$PORT)"
	@echo "make compose-down  stop the compose stack"
	@echo "make clean         remove build output, caches and rendered videos"

# --------------------------------------------------------------------------- #
# Bring the solution up
# --------------------------------------------------------------------------- #
up: check-env web-build
	@echo ""
	@echo "  ANN2SNN dashboard  ->  http://localhost:$(PORT)   (Ctrl-C to stop)"
	@echo ""
	exec $(PY) -m gunicorn -b $(HOST):$(PORT) --workers $(WORKERS) --timeout $(TIMEOUT) server.wsgi:application

check-env:
	@$(PY) -c "import flask, flask_cors, gunicorn, torch, sim_engine" 2>/dev/null \
		|| { echo "error: Python deps missing. Run 'make install' first."; exit 1; }

install:
	$(PY) -m pip install -e ".[dashboard,video,dev]"

test: test-engine test-server test-render

test-engine:
	$(PY) -m pytest tests -q

test-server:
	$(PY) -m pytest server/tests -q

test-render:
	$(PY) -m pytest tools/tests -q

web-install:
	@if [ ! -d web/node_modules ]; then \
		echo "installing frontend deps (npm ci)..."; \
		cd web && npm ci; \
	else \
		echo "frontend deps already installed (web/node_modules)"; \
	fi

web-build: web-install
	cd web && npm run build

serve: check-env
	exec $(PY) -m gunicorn -b $(HOST):$(PORT) --workers $(WORKERS) --timeout $(TIMEOUT) server.wsgi:application

serve-dev:
	PORT=$(PORT) $(PY) server/app.py

videos-train:
	$(PY) tools/generate_videos.py

videos-fast:
	$(PY) tools/generate_videos.py --no-train

videos: videos-train

# --- self-contained Docker image (for external users) --------------------- #
docker-build:
	docker build -t $(IMAGE) .

# Build + run the container, automatically picking the first free host port
# starting at $(PORT) (busy hosts often have 8080 taken). Streams the logs and
# removes the container on Ctrl-C.
docker-run: docker-build
	mkdir -p videos
	-@docker rm -f ann2snn-dashboard >/dev/null 2>&1 || true
	@port=$(PORT); ok=0; \
	while [ $$port -le $$(($(PORT) + 25)) ]; do \
		if docker run -d --name ann2snn-dashboard -p $$port:8080 \
			-v "$$PWD/videos:/app/videos" $(IMAGE) >/dev/null 2>&1; then \
			ok=1; echo ""; \
			echo "  ANN2SNN dashboard  ->  http://localhost:$$port"; \
			echo "  (Ctrl-C stops it)"; echo ""; \
			break; \
		fi; \
		docker rm -f ann2snn-dashboard >/dev/null 2>&1 || true; \
		port=$$((port + 1)); \
	done; \
	if [ $$ok -ne 1 ]; then \
		echo "error: no free host port in $$(($(PORT)))..$$(($(PORT) + 25)))"; \
		exit 1; \
	fi; \
	trap 'docker rm -f ann2snn-dashboard >/dev/null 2>&1 || true' INT TERM EXIT; \
	docker logs -f ann2snn-dashboard

# Bring the containerised solution up. This is the reliable path: plain
# `docker build` + `docker run` with automatic free-port selection (no compose
# and no buildx/buildkit required).
up-docker:
	mkdir -p videos
	@echo ">> docker build + docker run  (auto-picking a free host port from $(PORT))"
	@$(MAKE) docker-run PORT=$(PORT)

# Optional compose workflow. Note it uses ${PORT:-8080}: if that host port is
# busy, pass another one, e.g. `make compose-up PORT=9000`.
compose-up:
	mkdir -p videos
	@if docker compose version >/dev/null 2>&1; then \
		PORT=$(PORT) docker compose up --build; \
	elif command -v docker-compose >/dev/null 2>&1; then \
		PORT=$(PORT) docker-compose up --build; \
	else \
		echo "error: docker compose / docker-compose not found; use 'make up-docker'"; \
		exit 1; \
	fi

compose-down:
	-docker compose down >/dev/null 2>&1 || docker-compose down >/dev/null 2>&1 || true

docker-videos:
	mkdir -p videos
	docker run --rm -v "$$PWD/videos:/app/videos" $(IMAGE) \
		python3 tools/generate_videos.py --outdir /app/videos

docker-shell:
	docker run --rm -it --entrypoint sh $(IMAGE)

docker-stop:
	-docker rm -f ann2snn-dashboard

clean:
	rm -rf web/node_modules server/static videos/*.mp4 web/dist
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
