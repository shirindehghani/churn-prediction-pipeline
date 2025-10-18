FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY app /app/app

RUN mkdir -p /app/artifacts

ENV ARTIFACT_DIR=/app/artifacts
ENV DB_SCHEMA=public
ENV FEATURE_TABLE=final_features
ENV SEQ_LEN=6

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
