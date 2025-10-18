# app/main.py
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

# Optional torch (only needed when best model is torch_*)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# -----------------------------
# DB / App config (env overrides)
# -----------------------------
DB_USER = os.getenv("DB_USER", "sn_dehghani")
DB_PASS = os.getenv("DB_PASS", "sndi")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "8000")
DB_NAME = os.getenv("DB_NAME", "sn_dehghani")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")
FEATURE_TABLE = os.getenv("FEATURE_TABLE", "final_features")
TARGET_COL = os.getenv("TARGET_COL", "label")
KEY_COLS = ["user_id", "month_start"]

SEQ_LEN = int(os.getenv("SEQ_LEN", "6"))

# BASE_DIR resolves to .../repo-root/app (both locally and in the container)
BASE_DIR = Path(__file__).resolve().parent
# NOTE: ARTIFACT_DIR can be provided by env. We resolve robustly below.
ARTIFACT_DIR_ENV = os.getenv("ARTIFACT_DIR", "").strip()

TABLE_FULL = f"{DB_SCHEMA}.{FEATURE_TABLE}"
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# -----------------------------
# Globals (populated at startup)
# -----------------------------
MODEL = None
FEATURE_COLS: Optional[List[str]] = None
MODEL_NAME = "unknown"
MODEL_TYPE = "sklearn"  # sklearn | torch_mlp | torch_seq
THRESHOLD = 0.5
PREPROC = None
MODEL_PATH = None
DEVICE = "cpu"
ARTIFACT_DIR = None  # resolved absolute path used by the app

app = FastAPI(title="Churn API", version="1.2")

# -----------------------------
# Helper: resolve artifacts dir
# -----------------------------
REQUIRED_META = ["best_model_info.json", "feature_columns.json"]

def dir_has_required_files(p: Path) -> bool:
    return all((p / f).exists() for f in REQUIRED_META)

def resolve_artifact_dir() -> Path:
    """
    Pick the artifact directory in a robust order:
    1) Explicit ARTIFACT_DIR env (absolute or relative to CWD; '~' expanded)
    2) /artifacts (docker-compose volume mount)
    3) BASE_DIR/../notebooks/artifacts  (your repo layout)
    4) BASE_DIR/artifacts                (fallback)
    """
    candidates: List[Path] = []

    if ARTIFACT_DIR_ENV:
        # allow "~" and relative paths
        candidates.append(Path(os.path.expanduser(ARTIFACT_DIR_ENV)).resolve())

    candidates.append(Path("/artifacts"))  # compose mount
    candidates.append((BASE_DIR.parent / "notebooks" / "artifacts").resolve())
    candidates.append((BASE_DIR / "artifacts").resolve())

    tried = []
    for cand in candidates:
        tried.append(str(cand))
        if cand.is_dir() and dir_has_required_files(cand):
            return cand

    # Build a helpful error message
    msg = (
        "Could not locate the artifacts directory containing the required files.\n"
        f"Required: {', '.join(REQUIRED_META)}\n"
        "Tried (in order):\n  - " + "\n  - ".join(tried) + "\n\n"
        "Fix one of the following:\n"
        "  • Set ARTIFACT_DIR to the correct absolute path (env var), OR\n"
        "  • Mount your host artifacts to /artifacts in docker-compose, OR\n"
        "  • Ensure repo layout has notebooks/artifacts with the files."
    )
    raise RuntimeError(msg)

# -----------------------------
# Helper: resolve paths from best_model_info.json
# -----------------------------
def resolve_path_from_artifacts(p: str | None, artifacts_dir: Path) -> Path | None:
    """
    If p is absolute -> return Path(p).
    If p is relative -> return artifacts_dir / p.
    If p is None/empty -> return None.
    """
    if not p:
        return None
    pth = Path(p)
    return pth if pth.is_absolute() else (artifacts_dir / pth)

# -----------------------------
# DB helpers
# -----------------------------
def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True, future=True)

def fetch_user_rows(engine, user_id: int) -> pd.DataFrame | None:
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

# -----------------------------
# Torch models (match training)
# -----------------------------
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
        out, _ = self.rnn(x)
        h_last = out[:, -1, :]
        logits = self.head(h_last)
        return logits

# -----------------------------
# Feature alignment helpers
# -----------------------------
def align_and_order_columns(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df.drop(columns=[c for c in KEY_COLS + [TARGET_COL] if c in df.columns], errors="ignore")
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols]
    return X

def build_sequence_from_user_rows(df_user: pd.DataFrame, preproc, feature_cols: List[str], seq_len: int) -> np.ndarray:
    if df_user.empty:
        raise ValueError("No rows for user to build sequence.")
    df_u = df_user.sort_values("month_start").tail(seq_len)
    X_df = align_and_order_columns(df_u, feature_cols)
    X_t = preproc.transform(X_df)
    X_t = np.asarray(X_t, dtype=np.float32)
    k, feat = X_t.shape
    out = np.zeros((seq_len, feat), dtype=np.float32)
    out[-k:, :] = X_t
    return out[np.newaxis, :, :]

# -----------------------------
# Startup: load artifacts
# -----------------------------
@app.on_event("startup")
def load_artifacts():
    """Load model and metadata once at startup with clear error messages."""
    global MODEL, FEATURE_COLS, MODEL_NAME, MODEL_TYPE, THRESHOLD, PREPROC, MODEL_PATH, DEVICE, ARTIFACT_DIR

    # Pick and validate artifacts directory
    ARTIFACT_DIR = resolve_artifact_dir()

    # Load meta
    with open(ARTIFACT_DIR / "best_model_info.json") as f:
        info = json.load(f)
    MODEL_NAME = info.get("best_model", "unknown")
    THRESHOLD = float(info.get("threshold", 0.5))

    # Determine type from name
    name = (MODEL_NAME or "").lower()
    if "torch_lstm" in name or "torch_gru" in name:
        MODEL_TYPE = "torch_seq"
    elif "torch_mlp" in name:
        MODEL_TYPE = "torch_mlp"
    else:
        MODEL_TYPE = "sklearn"

    # Load feature columns
    with open(ARTIFACT_DIR / "feature_columns.json") as f:
        FEATURE_COLS = json.load(f)

    # Resolve any optional paths from the info (absolute-safe)
    MODEL_PATH = resolve_path_from_artifacts(info.get("model_path"), ARTIFACT_DIR)
    PREPROC_PATH_INFO = resolve_path_from_artifacts(info.get("preprocessor_path"), ARTIFACT_DIR)

    # Load model
    if MODEL_TYPE == "sklearn":
        pkl_path = MODEL_PATH or (ARTIFACT_DIR / "model_best.pkl")
        if not pkl_path.exists():
            raise RuntimeError(f"Missing sklearn model file at: {pkl_path}")
        MODEL = joblib.load(pkl_path)
        PREPROC = None

    else:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch model selected but PyTorch is not installed/available.")
        # pick device; in docker CPU is typical
        DEVICE = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )

        pt_path = MODEL_PATH or (ARTIFACT_DIR / "model_best_torch.pt")
        preproc_path = PREPROC_PATH_INFO or (ARTIFACT_DIR / "preprocessing_pipeline.pkl")
        if not pt_path.exists():
            raise RuntimeError(f"Missing torch model state_dict at: {pt_path}")
        if not preproc_path.exists():
            raise RuntimeError(f"Missing preprocessing pipeline at: {preproc_path}")

        PREPROC = joblib.load(preproc_path)

        # Infer in_dim from PREPROC
        dummy = pd.DataFrame({c: [np.nan] for c in FEATURE_COLS})
        in_dim = PREPROC.transform(dummy).shape[1]

        if MODEL_TYPE == "torch_mlp":
            MODEL = MLP(in_dim=in_dim)
            MODEL.load_state_dict(torch.load(pt_path, map_location="cpu"))
            MODEL.to(DEVICE).eval()
        else:
            kind = "lstm" if "lstm" in name else "gru"
            MODEL = RNNBinary(in_dim=in_dim, hidden=128, kind=kind, num_layers=1, dropout=0.0)
            MODEL.load_state_dict(torch.load(pt_path, map_location="cpu"))
            MODEL.to(DEVICE).eval()

# -----------------------------
# Endpoints
# -----------------------------
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

@app.get("/health")
def health():
    info = {
        "ok": True,
        "model": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "threshold": THRESHOLD,
        "artifacts_dir": str(ARTIFACT_DIR) if ARTIFACT_DIR else None,
        "db_url": DB_URL,
        "table_full": TABLE_FULL,
        "seq_len": SEQ_LEN,
    }
    try:
        insp = inspect(get_engine())
        info["table_exists"] = insp.has_table(FEATURE_TABLE, schema=DB_SCHEMA)
    except Exception as e:
        info["table_check_error"] = f"{type(e).__name__}: {e}"
    # Also list which required files exist (helps debugging)
    if ARTIFACT_DIR:
        info["artifacts_present"] = {
            "best_model_info.json": (ARTIFACT_DIR / "best_model_info.json").exists(),
            "feature_columns.json": (ARTIFACT_DIR / "feature_columns.json").exists(),
            "model_best.pkl": (ARTIFACT_DIR / "model_best.pkl").exists(),
            "model_best_torch.pt": (ARTIFACT_DIR / "model_best_torch.pt").exists(),
            "preprocessing_pipeline.pkl": (ARTIFACT_DIR / "preprocessing_pipeline.pkl").exists(),
        }
    return info

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    if MODEL is None or FEATURE_COLS is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded yet. Check server logs.")

    try:
        df_user = fetch_user_rows(get_engine(), body.user_id)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e.__class__.__name__}: {e}")

    if df_user is None or df_user.empty:
        raise HTTPException(404, f"user_id {body.user_id} not found in {TABLE_FULL}")

    month_str = None
    if "month_start" in df_user.columns and not df_user["month_start"].isna().all():
        month_str = pd.to_datetime(df_user["month_start"].max()).strftime("%Y-%m-%d")

    if MODEL_TYPE == "sklearn":
        latest = df_user.sort_values("month_start").tail(1)
        X = latest.drop(columns=[c for c in KEY_COLS + [TARGET_COL] if c in latest.columns], errors="ignore")
        for c in FEATURE_COLS:
            if c not in X.columns:
                X[c] = np.nan
        X = X[FEATURE_COLS]
        prob = float(MODEL.predict_proba(X)[:, 1][0])

    elif MODEL_TYPE == "torch_mlp":
        latest = df_user.sort_values("month_start").tail(1)
        X = align_and_order_columns(latest, FEATURE_COLS)
        X_t = PREPROC.transform(X)
        X_t = np.asarray(X_t, dtype=np.float32)
        with torch.no_grad():
            logits = MODEL(torch.tensor(X_t, dtype=torch.float32).to(DEVICE)).cpu().numpy().ravel()[0]
            prob = float(1.0 / (1.0 + np.exp(-logits)))

    else:  # torch_seq
        X_seq = build_sequence_from_user_rows(df_user, PREPROC, FEATURE_COLS, SEQ_LEN)
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
