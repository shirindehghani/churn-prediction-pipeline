FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps (for psycopg2 + science stack)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev curl \
 && rm -rf /var/lib/apt/lists/*

# Python deps (API + ML). Torch CPU so torch models work inside container.
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

# Copy only the app/ directory (your main.py is inside here)
COPY ./app /app/app

# Optional: keep repo root on import path, handy if you add packages later
ENV PYTHONPATH=/app

# Non-root
RUN useradd -m appuser
USER appuser

EXPOSE 8001
# NOTE: main.py is inside app/, so the module path is app.main:app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
