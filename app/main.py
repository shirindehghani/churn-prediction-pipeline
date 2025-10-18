# main.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# ---------- Env (override in docker/docker-compose/.env) ----------
DB_USER = os.getenv("DB_USER", "sn_dehghani")
DB_PASS = os.getenv("DB_PASS", "sndi")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "8000")  # non-standard PG port per your setup
DB_NAME = os.getenv("DB_NAME", "sn_dehghani")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")
FEATURE_TABLE = os.getenv("FEATURE_TABLE", "final_features")
TARGET_COL = os.getenv("TARGET_COL", "label")
KEY_COLS = ["user_id", "month_start"]

# Resolve artifacts relative to this file by default (project_root/notebooks/artifacts)
BASE_DIR = Path(__file__).resolve().parent               # .../app
DEFAULT_ARTIFACT_DIR = (BASE_DIR.parent / "notebooks" / "artifacts").as_posix()
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)

TABLE_FULL = f"{DB_SCHEMA}.{FEATURE_TABLE}"
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ---------- Globals populated at startup ----------
MODEL = None
FEATURE_COLS = None
MODEL_NAME = "unknown"
THRESHOLD = 0.5

# ---------- FastAPI ----------
app = FastAPI(title="Churn API", version="1.0")


class PredictIn(BaseModel):
    user_id: int


class PredictOut(BaseModel):
    user_id: int
    month_start: str | None
    probability: float
    will_churn: str
    threshold: float
    model_name: str


def get_engine():
    # We fully qualify the table name with schema, so we don't need to set search_path.
    return create_engine(DB_URL, pool_pre_ping=True, future=True)


def fetch_latest(engine, user_id: int) -> pd.DataFrame | None:
    q = text(f"""
        SELECT *
        FROM {TABLE_FULL}
        WHERE user_id = :uid
        ORDER BY month_start DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"uid": user_id})
    if df.empty:
        return None
    if "month_start" in df.columns:
        df["month_start"] = pd.to_datetime(df["month_start"])
    return df


@app.on_event("startup")
def load_artifacts():
    """Load model and metadata once at startup with clear error messages."""
    global MODEL, FEATURE_COLS, MODEL_NAME, THRESHOLD

    # Validate artifact presence first (gives nicer errors than a raw FileNotFoundError)
    needed = ["best_model_info.json", "feature_columns.json", "model_best.pkl"]
    missing = [f for f in needed if not Path(ARTIFACT_DIR, f).exists()]
    if missing:
        raise RuntimeError(
            "Artifact files missing.\n"
            f"ARTIFACT_DIR={ARTIFACT_DIR}\n"
            f"Missing: {', '.join(missing)}\n"
            "Fix: set ARTIFACT_DIR to the correct absolute path or place the files there."
        )

    # Load info
    with open(os.path.join(ARTIFACT_DIR, "best_model_info.json")) as f:
        info = json.load(f)
    MODEL_NAME = info.get("best_model", "unknown")
    THRESHOLD = float(info.get("threshold", 0.5))

    # Load feature columns
    with open(os.path.join(ARTIFACT_DIR, "feature_columns.json")) as f:
        FEATURE_COLS = json.load(f)

    # Load model (sklearn Pipeline / XGBoost inside)
    MODEL = joblib.load(os.path.join(ARTIFACT_DIR, "model_best.pkl"))


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_NAME,
        "threshold": THRESHOLD,
        "artifacts_dir": ARTIFACT_DIR,
    }


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    # Basic sanity checks
    if MODEL is None or FEATURE_COLS is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded yet. Check server logs."
        )

    try:
        row = fetch_latest(get_engine(), body.user_id)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e.__class__.__name__}: {e}")

    if row is None:
        raise HTTPException(404, f"user_id {body.user_id} not found")

    month_str = row["month_start"].iloc[0].strftime("%Y-%m-%d") if "month_start" in row.columns else None

    # Prepare features: all minus keys and label
    X = row.drop(columns=[c for c in KEY_COLS + [TARGET_COL] if c in row.columns], errors="ignore")

    # Align to training columns
    for c in FEATURE_COLS:
        if c not in X.columns:
            X[c] = np.nan
    X = X[FEATURE_COLS]

    # Predict probability
    prob = float(MODEL.predict_proba(X)[:, 1][0])
    label = "yes" if prob >= THRESHOLD else "no"

    return PredictOut(
        user_id=body.user_id,
        month_start=month_str,
        probability=prob,
        will_churn=label,
        threshold=THRESHOLD,
        model_name=MODEL_NAME,
    )
