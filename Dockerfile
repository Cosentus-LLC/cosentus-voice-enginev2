# syntax=docker/dockerfile:1.7

# Cosentus voice-engine v2 — Layer 10 container image.
#
# Two-stage build:
#
# Stage 1 (builder): install build-essential + uv, run ``uv pip install
#   --system .`` to resolve everything into the Python site-packages
#   tree. Drops in the local ``app`` package alongside Pipecat 1.1.0
#   and friends.
#
# Stage 2 (runtime): copy the populated site-packages + the python
#   binaries from the builder, install the minimum apt runtime deps
#   (``libgomp1`` for ONNX/Silero models, ``ca-certificates`` for
#   HTTPS to Daily/Bedrock/Lambda), create a non-root user, expose
#   port 8080, run ``python -m app.main``.
#
# Validated Layer 9.5 sizing for the matching Fargate task:
#   * 1 vCPU, 2 GB memory, ``stopTimeout=120 s``.
#
# Build (linux/amd64 for Fargate, from the repo root):
#
#   docker build --platform linux/amd64 \
#     -t cosentus-voice-engine:$(git rev-parse --short HEAD) \
#     -t cosentus-voice-engine:latest .
#
# Local smoke-test (env from the .env.skeleton):
#
#   docker run --rm -p 8080:8080 \
#     --env-file backend/voice-agent/scripts/.env.skeleton \
#     cosentus-voice-engine:latest
#
# Then ``curl http://localhost:8080/health`` ⇒ 200.
#
# ECR push lives in Layer 11 once the CDK provisions the repo.

# ── Stage 1: builder ──────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Build-time env: stop pip from caching wheels, force unbuffered output
# during ``uv pip install`` so the build log is informative.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time apt deps. ``build-essential`` covers gcc + make for any
# wheel that doesn't ship a manylinux build. ``ca-certificates`` lets
# pip / uv talk to PyPI over HTTPS in restricted networks. We don't
# pin versions — this is a build stage, drop on rebuild.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# uv is the same package manager the project uses locally — picks up
# pyproject.toml and resolves the lockfile when present. Faster than
# pip; consistent with developer workflow.
RUN pip install --no-cache-dir uv

WORKDIR /build

# Layer-cache strategy: copy ONLY the dependency manifests first, run
# the install, then copy the app code. ``pyproject.toml`` and
# ``uv.lock`` change less frequently than ``app/`` source, so the
# expensive deps install layer caches across most code changes.
#
# The local ``app`` package is referenced by setuptools via
# ``[tool.setuptools.packages.find] where = ["backend/voice-agent"]``,
# so we need a stub directory present at install time for setuptools
# to validate the package layout. Real source copied below.
COPY pyproject.toml ./
COPY uv.lock ./
COPY README.md ./
RUN mkdir -p backend/voice-agent/app \
 && printf '"""stub for builder layer cache; real code copied in next layer."""\n' \
        > backend/voice-agent/app/__init__.py
# FROZEN dependency install. ``uv export --frozen`` turns the committed
# uv.lock into a fully-pinned, hash-locked requirements set WITHOUT
# re-resolving against PyPI; ``uv pip install -r`` then installs exactly
# those versions into the system site-packages.
#
# Why this matters (regression 2026-05-28): the previous
# ``uv pip install .`` ignored uv.lock and re-resolved the pyproject
# range on every build. A fresh prod build silently pulled pipecat-ai
# 1.2.1 instead of the Wave-6-validated 1.1.0, which broke ElevenLabs
# TTS (1008 voice_settings WebSocket error) and produced a totally
# silent call. With --frozen, a rebuild months from now installs
# byte-identical dependency versions. See tech-debt entry 20.
#
# ``--no-emit-project`` excludes the local app package (installed
# separately below with --no-deps); ``--no-dev`` drops pytest et al.
RUN uv export --frozen --no-emit-project --no-dev --format requirements-txt -o /tmp/requirements.txt \
 && uv pip install --system --no-cache-dir -r /tmp/requirements.txt

# Copy real app source on top. This invalidates a small layer (the
# stub install metadata) but the deps are already cached.
COPY backend/voice-agent/app ./backend/voice-agent/app
# Reinstall just the local package so the actual app code lands in
# site-packages. ``--no-deps`` keeps the cached deps install intact;
# ``--force-reinstall`` ensures the stub gets fully replaced.
RUN uv pip install --system --no-cache-dir --no-deps --force-reinstall .

# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Runtime env. ``PYTHONUNBUFFERED=1`` is critical for CloudWatch log
# ingestion — without it, structlog JSON lines stay in stdio buffers
# until the container exits.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SERVICE_PORT=8080

# Runtime apt deps. ``libgomp1`` is the GNU OpenMP runtime; ONNX
# Runtime (smart-turn model) and Silero VAD link against it. Without
# it Pipecat boot fails with ``OSError: libgomp.so.1: cannot open
# shared object file``. ``ca-certificates`` for outbound TLS to
# Daily / Bedrock / Lambda / AssemblyAI / ElevenLabs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy the populated site-packages (Pipecat 1.1.0 + the local ``app``
# package + transitive deps) and the python entry-point binaries
# (``python``, ``pip``, ``uv`` is dropped — only need what's used at
# runtime).
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages

# Non-root user. UID 1000 is the conventional first non-system user
# and matches what most Fargate guides use. Owns the working
# directory but not site-packages — site-packages is read-only at
# runtime.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin engine \
 && mkdir -p /app \
 && chown engine:engine /app

USER engine
WORKDIR /app

EXPOSE 8080

# Healthcheck: pure-python, no curl install needed. Returns 0 when
# /health responds 200 within 2 s. Start period covers Pipecat's
# Smart-Turn / Silero model loads on cold start (~1–2 s observed in
# Layer 9.5 scenario a).
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u, sys; \
sys.exit(0 if u.urlopen('http://127.0.0.1:8080/health', timeout=2).status==200 else 1)"]

# Entry point. Layer 9 owns SIGTERM via Phase 2's ``shutdown_tasks``
# set + ``add_done_callback`` cleanup — so a Docker SIGTERM (from
# ``docker stop`` or Fargate's stopTimeout-driven shutdown) flows
# through the same code path that the scale-test scenario e
# validated empirically at 5.0 s drain for 6 active sessions.
CMD ["python", "-m", "app.main"]
