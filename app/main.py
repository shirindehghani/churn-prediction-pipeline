# main.py
import os, json, joblib, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# ----- Env (override in docker-compose) -----
DB_USER = os.getenv("DB_USER", "sn_dehghani")
DB_PASS = os.getenv("DB_PASS", "sndi")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "sn_dehghani")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")
FEATURE_TABLE = os.getenv("FEATURE_TABLE", "final_features")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "./notebooks/artifacts")
TARGET_COL = os.getenv("TARGET_COL", "label")
KEY_COLS = ["user_id", "month_start"]
TABLE_FULL = f"{DB_SCHEMA}.{FEATURE_TABLE}"

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ----- Load artifacts once -----
with open(os.path.join(ARTIFACT_DIR, "best_model_info.json")) as f:
    info = json.load(f)
MODEL_NAME = info.get("best_model", "unknown")
THRESHOLD = float(info.get("threshold", 0.5))

with open(os.path.join(ARTIFACT_DIR, "feature_columns.json")) as f:
    FEATURE_COLS = json.load(f)

MODEL = joblib.load(os.path.join(ARTIFACT_DIR, "model_best.pkl"))  # sklearn Pipeline (XGBoost)

# ----- FastAPI -----
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
    eng = create_engine(DB_URL, pool_pre_ping=True, future=True)
    with eng.connect() as c:
        c.execute(text(f"SET search_path TO {DB_SCHEMA};"))
    return eng

def fetch_latest(engine, user_id: int) -> pd.DataFrame | None:
    q = text(f"""
        SELECT * FROM {TABLE_FULL}
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

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "threshold": THRESHOLD}

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    row = fetch_latest(get_engine(), body.user_id)
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
