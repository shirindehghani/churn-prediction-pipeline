from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlalchemy as sa
import pandas as pd, numpy as np, joblib, os, re
from datetime import datetime

DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://postgres:postgres@db:5432/postgres")
engine = sa.create_engine(DB_URL)

# Load model once
pipe = joblib.load('/app/model/model.pkl')
meta = joblib.load('/app/model/model_meta.pkl')
FEATURES = meta['features']
THRESH = meta['threshold']

app = FastAPI(title="Churn API", version="1.0")

class PredictResponse(BaseModel):
    user_id: int
    will_churn: bool
    probability: float

def fetch_user_month_latest(user_id: int) -> pd.DataFrame:
    """
    Compute the latest available user-month features on the fly to mirror training.
    We reuse the SQL definition from user_month_features but filter user_id and latest month.
    """
    q = """
    WITH latest AS (
      SELECT MAX(month_date) AS month_date
      FROM user_month_features
      WHERE user_id = :uid
    )
    SELECT *
    FROM user_month_features
    WHERE user_id = :uid
      AND month_date = (SELECT month_date FROM latest)
    """
    df = pd.read_sql(sa.text(q), engine, params={'uid': user_id})
    if df.empty:
        # fallback: try to compute minimal row from base orders if user exists
        exists = pd.read_sql("SELECT 1 FROM orders WHERE user_id=:uid LIMIT 1", engine, params={'uid': user_id})
        if exists.empty:
            raise HTTPException(status_code=404, detail="user_id not found")
        else:
            raise HTTPException(status_code=409, detail="No feature row for latest month; run features job.")
    return df

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/predict", response_model=PredictResponse)
def predict(user_id: int):
    df = fetch_user_month_latest(user_id)
    X = df[FEATURES].fillna(0)
    p = float(pipe.predict_proba(X)[:,1][0])
    out = PredictResponse(user_id=user_id, will_churn=bool(p >= THRESH), probability=p)
    return out
