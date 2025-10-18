# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for psycopg2 and numpy/pandas builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev curl \
 && rm -rf /var/lib/apt/lists/*

# If you have a requirements.txt, great. Otherwise inline install below.
COPY requirements.txt /app/requirements.txt

# ----- If you don't have a requirements.txt, comment the line above
# and uncomment the block below
# RUN pip install --no-cache-dir \
#     fastapi uvicorn[standard] \
#     sqlalchemy psycopg2-binary \
#     pandas numpy scikit-learn joblib \
#     # torch is optional; install only if you need torch_* models
#     # torch==2.4.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
#     && true

RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your app code
# Tree on host:
# repo-root/
#   app/
#     main.py
#   notebooks/
#     artifacts/...
COPY app/ /app/app/

# --------- OPTION A (volume mount; recommended) ----------
# Do NOT copy artifacts here. We’ll mount them at runtime to /artifacts.
# Leave the next line commented:
# COPY notebooks/artifacts/ /artifacts/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
