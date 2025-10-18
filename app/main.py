# main.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

# Optional torch imports (only needed for torch models)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

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

# Sequence length used in training (default 6). Keep in sync with your train code.
SEQ_LEN = int(os.getenv("SEQ_LEN", "6"))

# Resolve artifacts (defaults to project_root/notebooks/artifacts)
BASE_DIR = Path(__file__).resolve().parent               # .../app
DEFAULT_ARTIFACT_DIR = (BASE_DIR.parent / "notebooks" / "artifacts").as_posix()
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)

TABLE_FULL = f"{DB_SCHEMA}.{FEATURE_TABLE}"
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ---------- Globals populated at startup ----------
MODEL = None                 # sklearn Pipeline OR torch.nn.Module
FEATURE_COLS: Optional[List[str]] = None
MODEL_NAME = "unknown"       # e.g., "xgboost", "torch_lstm", etc.
MODEL_TYPE = "sklearn"       # "sklearn" | "torch_mlp" | "torch_seq"
THRESHOLD = 0.5
PREPROC = None               # sklearn ColumnTransformer for torch models
MODEL_PATH = None            # path to model file (for torch)
DEVICE = "cpu"               # torch device if used

# ---------- FastAPI ----------
app = FastAPI(title="Churn API", version="1.1")


class PredictIn(BaseModel):
    user_id: int


class PredictOut(BaseModel):
    user_id: int
    month_start: str | None
    probability: float
    will_churn: str
    threshold: float
    model_name: str
    model_type: str


def get_engine():
    # Fully qualify table names with schema; no need to set search_path.
    return create_engine(DB_URL, pool_pre_ping=True, future=True)


def fetch_user_rows(engine, user_id: int) -> pd.DataFrame | None:
    """
    Fetch *all* rows for user ordered by month_start ASC (for seq models),
    or at least the latest row (sklearn/mlp). We fetch all and slice later.
    """
    q = text(f"""
        SELECT *
        FROM {TABLE_FULL}
        WHERE user_id = :uid
        ORDER BY month_start ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"uid": user_id})
    if df.empty:
        return None
    if "month_start" in df.columns:
        df["month_start"] = pd.to_datetime(df["month_start"])
    return df


# ---------------------------
# Torch model definitions (must match training)
# ---------------------------
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
    def forward(self, x):
        return self.net(x)

class RNNBinary(nn.Module):
    def __init__(self, in_dim: int, hidden: int, kind: str = "lstm", num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        assert kind in ("lstm", "gru")
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

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.rnn(x)          # (B, T, H)
        h_last = out[:, -1, :]        # last time-step
        logits = self.head(h_last)    # (B, 1)
        return logits


# ---------------------------
# Helpers for building inputs
# ---------------------------
def align_and_order_columns(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Ensure df has all FEATURE_COLS in correct order, fill missing with NaN."""
    X = df.drop(columns=[c for c in KEY_COLS + [TARGET_COL] if c in df.columns], errors="ignore")
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols]
    return X

def build_sequence_from_user_rows(df_user: pd.DataFrame, preproc, feature_cols: List[str], seq_len: int) -> np.ndarray:
    """
    Transform user's rows and build a single padded sequence (1, T, F).
    - Use the **most recent** seq_len rows.
    - Left-pad with zeros if fewer than seq_len rows exist.
    """
    if df_user.empty:
        raise ValueError("No rows for user to build sequence.")

    # Most recent seq_len rows, keep ascending order for time
    df_u = df_user.sort_values("month_start").tail(seq_len)
    X_df = align_and_order_columns(df_u, feature_cols)
    X_t = preproc.transform(X_df)  # shape (k, F)
    X_t = np.asarray(X_t, dtype=np.float32)
    k, feat = X_t.shape

    # Pad to (seq_len, feat) with zeros at the front
    out = np.zeros((seq_len, feat), dtype=np.float32)
    out[-k:, :] = X_t
    # Add batch dimension -> (1, T, F)
    return out[np.newaxis, :, :]


# ---------------------------
# Startup: load artifacts and init model
# ---------------------------
@app.on_event("startup")
def load_artifacts():
    """Load model and metadata once at startup with clear error messages."""
    global MODEL, FEATURE_COLS, MODEL_NAME, MODEL_TYPE, THRESHOLD, PREPROC, MODEL_PATH, DEVICE

    needed = ["best_model_info.json", "feature_columns.json"]
    missing = [f for f in needed if not Path(ARTIFACT_DIR, f).exists()]
    # model file name varies by type; we'll validate after reading info
    if missing:
        raise RuntimeError(
            "Artifact files missing.\n"
            f"ARTIFACT_DIR={ARTIFACT_DIR}\n"
            f"Missing: {', '.join(missing)}\n"
            "Fix: set ARTIFACT_DIR to the correct absolute path or place the files there."
        )

    # Load meta info
    with open(os.path.join(ARTIFACT_DIR, "best_model_info.json")) as f:
        info = json.load(f)
    MODEL_NAME = info.get("best_model", "unknown")
    THRESHOLD = float(info.get("threshold", 0.5))
    MODEL_PATH = info.get("model_path")

    # Detect model type
    name = (MODEL_NAME or "").lower()
    if "torch_lstm" in name or "torch_gru" in name:
        MODEL_TYPE = "torch_seq"
    elif "torch_mlp" in name:
        MODEL_TYPE = "torch_mlp"
    else:
        MODEL_TYPE = "sklearn"

    # Load feature columns
    with open(os.path.join(ARTIFACT_DIR, "feature_columns.json")) as f:
        FEATURE_COLS = json.load(f)

    # Load model and (if needed) preprocessor
    if MODEL_TYPE == "sklearn":
        # Expect a sklearn pipeline with preprocessing inside
        pkl_path = MODEL_PATH or os.path.join(ARTIFACT_DIR, "model_best.pkl")
        if not Path(pkl_path).exists():
            raise RuntimeError(f"Missing sklearn model file at: {pkl_path}")
        MODEL = joblib.load(pkl_path)
        PREPROC = None
    else:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch model selected but PyTorch is not installed/available.")
        DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

        pt_path = MODEL_PATH or os.path.join(ARTIFACT_DIR, "model_best_torch.pt")
        preproc_path = info.get("preprocessor_path") or os.path.join(ARTIFACT_DIR, "preprocessing_pipeline.pkl")
        if not Path(pt_path).exists():
            raise RuntimeError(f"Missing torch model state_dict at: {pt_path}")
        if not Path(preproc_path).exists():
            raise RuntimeError(f"Missing preprocessing pipeline at: {preproc_path}")

        PREPROC = joblib.load(preproc_path)

        # Infer input dimension from PREPROC.transform output
        # Build a tiny dummy row with all feature columns
        dummy = pd.DataFrame({c: [np.nan] for c in FEATURE_COLS})
        in_dim = PREPROC.transform(dummy).shape[1]

        if MODEL_TYPE == "torch_mlp":
            MODEL = MLP(in_dim=in_dim)
            MODEL.load_state_dict(torch.load(pt_path, map_location="cpu"))
            MODEL.to(DEVICE).eval()
        else:  # torch_seq
            kind = "lstm" if "lstm" in name else "gru"
            MODEL = RNNBinary(in_dim=in_dim, hidden=128, kind=kind, num_layers=1, dropout=0.0)
            MODEL.load_state_dict(torch.load(pt_path, map_location="cpu"))
            MODEL.to(DEVICE).eval()


@app.get("/health")
def health():
    info = {
        "ok": True,
        "model": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "threshold": THRESHOLD,
        "artifacts_dir": ARTIFACT_DIR,
        "db_url": DB_URL,
        "table_full": TABLE_FULL,
        "seq_len": SEQ_LEN,
    }
    try:
        insp = inspect(get_engine())
        info["table_exists"] = insp.has_table(FEATURE_TABLE, schema=DB_SCHEMA)
    except Exception as e:
        info["table_check_error"] = f"{type(e).__name__}: {e}"
    return info


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    if MODEL is None or FEATURE_COLS is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded yet. Check server logs."
        )

    # Fetch user's rows (ascending by month)
    try:
        df_user = fetch_user_rows(get_engine(), body.user_id)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e.__class__.__name__}: {e}")

    if df_user is None or df_user.empty:
        raise HTTPException(404, f"user_id {body.user_id} not found in {TABLE_FULL}")

    # Latest month string for response
    month_str = None
    if "month_start" in df_user.columns and not df_user["month_start"].isna().all():
        month_str = pd.to_datetime(df_user["month_start"].max()).strftime("%Y-%m-%d")

    # ---- Run model by type ----
    if MODEL_TYPE == "sklearn":
        # Use latest row only (pipeline includes preprocessing)
        latest = df_user.sort_values("month_start").tail(1)
        X = latest.drop(columns=[c for c in KEY_COLS + [TARGET_COL] if c in latest.columns], errors="ignore")
        # Align to training columns (in case of drift)
        for c in FEATURE_COLS:
            if c not in X.columns:
                X[c] = np.nan
        X = X[FEATURE_COLS]
        prob = float(MODEL.predict_proba(X)[:, 1][0])

    elif MODEL_TYPE == "torch_mlp":
        # Latest row only, but we need PREPROC
        latest = df_user.sort_values("month_start").tail(1)
        X = align_and_order_columns(latest, FEATURE_COLS)
        X_t = PREPROC.transform(X)
        X_t = np.asarray(X_t, dtype=np.float32)
        with torch.no_grad():
            logits = MODEL(torch.tensor(X_t, dtype=torch.float32).to(DEVICE)).cpu().numpy().ravel()[0]
            prob = float(1.0 / (1.0 + np.exp(-logits)))

    else:  # "torch_seq"
        # Build padded sequence from user's history
        X_seq = build_sequence_from_user_rows(df_user, PREPROC, FEATURE_COLS, SEQ_LEN)  # (1, T, F)
        with torch.no_grad():
            logits = MODEL(torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)).cpu().numpy().ravel()[0]
            prob = float(1.0 / (1.0 + np.exp(-logits)))

    label = "yes" if prob >= THRESHOLD else "no"

    return PredictOut(
        user_id=body.user_id,
        month_start=month_str,
        probability=prob,
        will_churn=label,
        threshold=THRESHOLD,
        model_name=MODEL_NAME,
        model_type=MODEL_TYPE,
    )
