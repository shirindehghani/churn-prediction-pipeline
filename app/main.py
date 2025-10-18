import os
import json
import math
from typing import Optional, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

import joblib

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")
TABLE = os.getenv("FEATURE_TABLE", "final_features")
SEQ_LEN = int(os.getenv("SEQ_LEN", "6"))

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "/app/artifacts")
BEST_INFO_PATH = os.path.join(ARTIFACT_DIR, "best_model_info.json")
MODEL_PKL_PATH = os.path.join(ARTIFACT_DIR, "model_best.pkl")
MODEL_TORCH_PATH = os.path.join(ARTIFACT_DIR, "model_best_torch.pt")
PREPROC_PATH = os.path.join(ARTIFACT_DIR, "preprocessing_pipeline.pkl")
FEATURE_COLS_PATH = os.path.join(ARTIFACT_DIR, "feature_columns.json")

TARGET_COL = "label"
KEY_COLS = ["user_id", "month_start"]

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

app = FastAPI(title="Churn Prediction API", version="1.0.0")

class PredictIn(BaseModel):
    user_id: str

class PredictOut(BaseModel):
    user_id: str
    will_churn: Literal["yes", "no"]
    probability: float
    threshold: float
    model: str


def get_engine():
    return create_engine(DB_URL)

def load_artifacts():
    if not os.path.exists(BEST_INFO_PATH):
        raise RuntimeError(f"Missing {BEST_INFO_PATH}. Did you copy artifacts from training?")
    if not os.path.exists(FEATURE_COLS_PATH):
        raise RuntimeError(f"Missing {FEATURE_COLS_PATH}.")
    with open(BEST_INFO_PATH, "r") as f:
        best_info = json.load(f)
    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)

    best_model_name = best_info.get("best_model")
    threshold = best_info.get("threshold", 0.5)

    if best_model_name in ("logreg", "random_forest", "xgboost"):
        if not os.path.exists(MODEL_PKL_PATH):
            raise RuntimeError("Expected sklearn model file at artifacts/model_best.pkl")
        model = joblib.load(MODEL_PKL_PATH)
        preproc = None
        model_type = "sklearn"
    elif best_model_name in ("torch_mlp", "torch_lstm", "torch_gru"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available but the best model is a torch model.")
        if not os.path.exists(MODEL_TORCH_PATH):
            raise RuntimeError("Expected torch model file at artifacts/model_best_torch.pt")
        if not os.path.exists(PREPROC_PATH):
            raise RuntimeError("Expected preprocessing pipeline at artifacts/preprocessing_pipeline.pkl")
        preproc = joblib.load(PREPROC_PATH)
        if best_model_name == "torch_mlp":
            model = _load_mlp_model_state(MODEL_TORCH_PATH, in_dim=None)
        else:
            model = _build_seq_model_from_state(MODEL_TORCH_PATH, in_dim=None, hidden=128,
                                                kind="lstm" if best_model_name == "torch_lstm" else "gru")
        model_type = "torch_seq" if best_model_name in ("torch_lstm", "torch_gru") else "torch_mlp"
    else:
        raise RuntimeError(f"Unknown best_model: {best_model_name}")

    return {
        "best_model_name": best_model_name,
        "threshold": float(threshold),
        "feature_cols": feature_cols,
        "model": model,
        "preproc": preproc,
        "model_type": model_type,
    }

class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

class RNNBinary(nn.Module):
    def __init__(self, in_dim: int, hidden: int, kind: str = "lstm", num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        if kind == "lstm":
            self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        else:
            self.rnn = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.rnn(x)
        h_last = out[:, -1, :]
        return self.head(h_last)

def _load_mlp_model_state(path: str, in_dim: Optional[int]) -> nn.Module:
    if in_dim is None:
        tmp = MLP(1)
        sd = torch.load(path, map_location="cpu")
        tmp.load_state_dict({k: v for k, v in sd.items() if k in tmp.state_dict() and tmp.state_dict()[k].shape == v.shape}, strict=False)
        return tmp
    m = MLP(in_dim)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m

def _build_seq_model_from_state(path: str, in_dim: Optional[int], hidden: int, kind: str) -> nn.Module:
    if in_dim is None:
        tmp = RNNBinary(1, hidden=hidden, kind=kind)
        sd = torch.load(path, map_location="cpu")
        tmp.load_state_dict({k: v for k, v in sd.items() if k in tmp.state_dict() and tmp.state_dict()[k].shape == v.shape}, strict=False)
        return tmp
    m = RNNBinary(in_dim=in_dim, hidden=hidden, kind=kind)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _sklearn_predict_proba(model, X_df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X_df)[:, 1]

def _torch_mlp_predict(model, preproc, X_df: pd.DataFrame) -> np.ndarray:
    X = preproc.transform(X_df)
    if isinstance(model, MLP) and next(model.parameters()).shape[0] != X.shape[1]:
        m = MLP(X.shape[1])
        m.load_state_dict(torch.load(MODEL_TORCH_PATH, map_location="cpu"))
        model = m
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).numpy().ravel()
    return _sigmoid(logits)

def _build_sequence_matrix(preproc, df_user: pd.DataFrame, feature_cols: list, seq_len: int) -> np.ndarray:
    X = preproc.transform(df_user[feature_cols])
    n_feat = X.shape[1]
    out = np.zeros((1, seq_len, n_feat), dtype=np.float32)
    w = X[-seq_len:]
    pad = seq_len - w.shape[0]
    if pad > 0:
        out[0, :pad, :] = 0.0
        out[0, pad:, :] = w
    else:
        out[0, :, :] = w
    return out

def _torch_seq_predict(model, preproc, df_user_seq: pd.DataFrame, feature_cols: list, seq_len: int) -> np.ndarray:
    Xseq = _build_sequence_matrix(preproc, df_user_seq, feature_cols, seq_len)
    in_dim = Xseq.shape[2]
    if isinstance(model, RNNBinary) and model.rnn.input_size != in_dim:
        kind = "lstm" if isinstance(model.rnn, torch.nn.modules.rnn.LSTM) else "gru"
        m = RNNBinary(in_dim=in_dim, hidden=128, kind=kind)
        m.load_state_dict(torch.load(MODEL_TORCH_PATH, map_location="cpu"))
        model = m
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xseq, dtype=torch.float32)).numpy().ravel()
    return _sigmoid(logits)

def fetch_user_rows(engine, user_id: str) -> pd.DataFrame:
    with engine.connect() as conn:
        conn.execute(text(f"SET search_path TO {DB_SCHEMA};"))
        q = text(f"""
            SELECT * FROM {DB_SCHEMA}.{TABLE}
            WHERE user_id = :uid
            ORDER BY month_start ASC
        """)
        df = pd.read_sql(q, conn, params={"uid": user_id})
    if "month_start" in df.columns:
        df["month_start"] = pd.to_datetime(df["month_start"])
    return df

def latest_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values("month_start").tail(1)

ART = load_artifacts()
ENGINE = get_engine()

@app.get("/healthz")
def health():
    return {"status": "ok", "model": ART["best_model_name"]}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    user_id = payload.user_id
    df_user = fetch_user_rows(ENGINE, user_id)
    if df_user.empty:
        raise HTTPException(status_code=404, detail=f"No rows found for user_id={user_id}")

    feature_cols = ART["feature_cols"]
    missing = [c for c in feature_cols if c not in df_user.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Feature columns missing from table: {missing}")

    best = ART["best_model_name"]
    threshold = ART["threshold"]
    model_type = ART["model_type"]

    if model_type == "sklearn":
        X = latest_row(df_user)[feature_cols]
        prob = float(_sklearn_predict_proba(ART["model"], X)[0])
    elif model_type == "torch_mlp":
        X = latest_row(df_user)[feature_cols]
        prob = float(_torch_mlp_predict(ART["model"], ART["preproc"], X)[0])
    else:
        if df_user.shape[0] == 1:
            pass
        prob = float(_torch_seq_predict(ART["model"], ART["preproc"], df_user, feature_cols, SEQ_LEN)[0])

    will_churn = "yes" if prob >= threshold else "no"
    return PredictOut(
        user_id=user_id,
        will_churn=will_churn,
        probability=round(prob, 6),
        threshold=round(float(threshold), 6),
        model=best,
    )
