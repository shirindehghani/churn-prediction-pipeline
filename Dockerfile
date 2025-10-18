FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps:
# - libpq-dev: psycopg2
# - build-essential,gcc: wheels fallback/compilation (safe to keep)
# - curl: useful for container healthchecks
# - libgomp1: REQUIRED by xgboost runtime (libgomp.so.1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Python deps (Torch CPU so torch models can run in container)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    numpy \
    pandas \
    scikit-learn \
    joblib \
    "SQLAlchemy>=2.0" \
    psycopg2-binary \
    pydantic \
    xgboost \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

# Copy your API package (main.py is inside app/)
COPY ./app /app/app

# Optional: keep repo root on import path (handy if you add local packages later)
ENV PYTHONPATH=/app

# Fail fast if the module path is wrong (build will error immediately)
RUN python - <<'PY'
import importlib
import sys
m = importlib.import_module('app.main')
print("OK: imported", m.__name__)
PY

# Non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 8001

# main.py lives in app/, so use app.main:app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
