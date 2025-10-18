# Use Python 3.10 (works with your type hints and libs)
FROM python:3.10-slim

# Helpful runtime settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# XGBoost needs libgomp, keep image slim otherwise
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies (add xgboost because your model pipeline mentions it)
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    sqlalchemy psycopg2-binary \
    pandas numpy scikit-learn joblib xgboost

# Copy your app code (assumes your main.py lives in ./app/)
COPY app/ /app/app/

# Expose API port (container-side)
EXPOSE 8001

# Run the API (bind to all interfaces for container networking)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
