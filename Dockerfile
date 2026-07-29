# FastAPI kernel backend.
#
# No API keys are baked in: kernel/config.yaml is mounted at runtime by
# docker-compose (see that file), and the image works with no config at all —
# it falls back to the offline hashing embedder exactly as a fresh clone does.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so edits to source do not invalidate the pip layer.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY kernel/ ./kernel/
COPY api/ ./api/
COPY agents/ ./agents/
COPY shell/ ./shell/
COPY benchmarks/ ./benchmarks/

# Writable state. Both are gitignored on the host and live on named volumes in
# compose so they survive restarts without ending up in the image.
RUN mkdir -p /app/chroma_db /app/fs_root

EXPOSE 8000

# curl is used by compose's healthcheck; keep it in the runtime layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
