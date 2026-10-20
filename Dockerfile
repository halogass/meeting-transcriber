# Meeting Transcriber container.
#
# GPU (default):  docker compose up --build
# CPU only:       docker compose --profile cpu up --build meeting-transcriber-cpu
#
# In a container the app is mainly for uploaded recordings. Live recording also works on a
# Linux host if you share the PulseAudio/PipeWire socket (see docker-compose.yml).

ARG BUILD_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
ARG RUNTIME_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04

# ---------------------------------------------------------------- whisper.cpp build
FROM ${BUILD_IMAGE} AS whisper
ARG FORCE_CPU=
# Turing (GTX 16xx / RTX 20xx) through Ada (RTX 40xx); override with --build-arg CUDA_ARCHS=...
ARG CUDA_ARCHS="75;80;86;89"
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY scripts/build_whisper.sh scripts/build_whisper.sh
RUN FORCE_CPU="${FORCE_CPU}" CUDA_ARCHS="${CUDA_ARCHS}" scripts/build_whisper.sh \
    && rm -rf vendor

# ---------------------------------------------------------------- runtime
FROM ${RUNTIME_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    MT_HOST=0.0.0.0 \
    MT_PORT=8765
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv ffmpeg pulseaudio-utils libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --python /usr/bin/python3 && rm -rf /root/.cache/uv
COPY --from=whisper /src/bin/ /app/bin/
COPY app/ app/
COPY scripts/ scripts/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh scripts/*.sh

VOLUME ["/app/data", "/app/models"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://127.0.0.1:8765/api/status >/dev/null || exit 1
ENTRYPOINT ["/entrypoint.sh"]
