from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
SCHEMA  = "public"

ROW_LIMIT = None

MODEL_NAME = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
BATCH_SIZE = 64
MAX_LEN    = 256


engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    future=True
)

features_sql = """
DROP TABLE IF EXISTS public.features CASCADE;

WITH
orders_clean AS (
  SELECT
    user_id,
    order_id,
    CASE
      WHEN NULLIF(order_date,'') ~ '^\d{4}-\d{2}-\d{2}([ T]\\d{2}:\\d{2}(:\\d{2})?)?$'
      THEN (order_date)::timestamp ELSE NULL END AS order_ts,
    CASE
      WHEN lower(coalesce(is_otd,'')) IN ('1','true','t','yes','y') THEN 1
      WHEN lower(coalesce(is_otd,'')) IN ('0','false','f','no','n')  THEN 0
      ELSE NULL END AS is_otd_flag,
    lower(NULLIF(delivery_status,'')) AS delivery_status
  FROM public.orders
  WHERE coalesce(user_id,'')<>'' AND NULLIF(order_date,'')<>''
),
orders_month AS (
  SELECT
    user_id,
    order_id,
    date_trunc('month', order_ts)::date AS month_start,
    order_ts,
    is_otd_flag,
    delivery_status
  FROM orders_clean
  WHERE order_ts IS NOT NULL
),
orders_agg AS (
  SELECT
    user_id,
    month_start,
    COUNT(*)                                 AS orders_cnt_m,
    AVG(is_otd_flag::float)                  AS otd_rate_m,
    1.0 - AVG(is_otd_flag::float)            AS late_rate_m,
    AVG(CASE WHEN delivery_status='cancelled' THEN 1 ELSE 0 END) AS cancel_rate_m,
    AVG(CASE WHEN delivery_status='returned'  THEN 1 ELSE 0 END) AS return_rate_m,
    MAX(order_ts)                            AS last_order_ts_m
  FROM orders_month
  GROUP BY user_id, month_start
),
first_last AS (
  SELECT
    user_id,
    MIN(month_start) AS first_month
  FROM orders_agg
  GROUP BY user_id
),
crm_clean AS (
  SELECT
    c.order_id,
    NULLIF(c.crm_delivery_request_count,'')::int AS crm_req_cnt,
    NULLIF(c.crm_fake_delivery_request_count,'')::int AS crm_fake_cnt,
    NULLIF(c.rate_to_shop,'')::int    AS customer_rate,
    NULLIF(c.rate_to_courier,'')::int AS courier_rate
  FROM public.crm c
),
crm_join AS (
  SELECT
    om.user_id,
    om.month_start,
    cc.crm_req_cnt,
    cc.crm_fake_cnt,
    cc.customer_rate,
    cc.courier_rate
  FROM orders_month om
  JOIN crm_clean cc USING (order_id)
),
crm_agg AS (
  SELECT
    user_id,
    month_start,
    COALESCE(SUM(crm_req_cnt), 0)  AS crm_req_cnt_sum_m,
    COALESCE(SUM(crm_fake_cnt), 0) AS crm_fake_cnt_sum_m,
    AVG(customer_rate)             AS customer_rate_avg_m,
    AVG(courier_rate)              AS courier_rate_avg_m
  FROM crm_join
  GROUP BY user_id, month_start
),
comments_join AS (
  SELECT
    om.user_id,
    om.month_start,
    lower(coalesce(cm.description,'')) AS desc_lc
  FROM public.comments cm
  JOIN orders_month om USING (order_id)
),
comments_agg AS (
  SELECT
    user_id,
    month_start,
    COUNT(*) AS comments_cnt_m,
    AVG(length(desc_lc))::float AS comment_avg_len_m
  FROM comments_join
  GROUP BY user_id, month_start
),
base AS (
  SELECT
    oa.user_id,
    oa.month_start,
    oa.orders_cnt_m,
    oa.otd_rate_m,
    oa.late_rate_m,
    oa.cancel_rate_m,
    oa.return_rate_m,
    COALESCE(cr.crm_req_cnt_sum_m, 0)  AS crm_req_cnt_sum_m,
    COALESCE(cr.crm_fake_cnt_sum_m, 0) AS crm_fake_cnt_sum_m,
    cr.customer_rate_avg_m,
    cr.courier_rate_avg_m,
    COALESCE(cm.comments_cnt_m, 0)     AS comments_cnt_m,
    cm.comment_avg_len_m,
    oa.last_order_ts_m,
    fl.first_month
  FROM orders_agg oa
  LEFT JOIN crm_agg      cr USING (user_id, month_start)
  LEFT JOIN comments_agg cm USING (user_id, month_start)
  JOIN first_last fl USING (user_id)
),
roll AS (
  SELECT
    b.*,
    SUM(orders_cnt_m) OVER (PARTITION BY user_id ORDER BY month_start
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS orders_cnt_3m,
    (EXTRACT(YEAR FROM age(month_start, first_month))*12
     + EXTRACT(MONTH FROM age(month_start, first_month)))::int AS tenure_months,
    ((month_start + INTERVAL '1 month' - INTERVAL '1 day')::date - last_order_ts_m::date)::int AS recency_days
  FROM base b
),
feat AS (
  SELECT
    r.*,
    CASE WHEN r.orders_cnt_m > 0 THEN r.crm_req_cnt_sum_m::float / r.orders_cnt_m ELSE NULL END AS crm_req_per_order_m,
    (r.late_rate_m * r.orders_cnt_m) AS late_x_orders_m
  FROM roll r
),
lbl AS (
  SELECT
    f.user_id,
    f.month_start,
    CASE WHEN COALESCE(n.orders_cnt_m, 0) = 0 THEN 1 ELSE 0 END AS label
  FROM feat f
  LEFT JOIN feat n
    ON n.user_id = f.user_id
   AND n.month_start = (f.month_start + INTERVAL '1 month')::date
)
SELECT
  f.user_id,
  f.month_start,
  f.orders_cnt_m,
  f.orders_cnt_3m,
  f.otd_rate_m,
  f.late_rate_m,
  f.cancel_rate_m,
  f.return_rate_m,
  f.crm_req_cnt_sum_m,
  f.crm_fake_cnt_sum_m,
  f.customer_rate_avg_m,
  f.courier_rate_avg_m,
  f.crm_req_per_order_m,
  f.comments_cnt_m,
  f.comment_avg_len_m,
  f.tenure_months,
  f.recency_days,
  f.late_x_orders_m,
  l.label
INTO public.features
FROM feat f
JOIN lbl l USING (user_id, month_start);

CREATE INDEX IF NOT EXISTS ix_features_user_month
ON public.features (user_id, month_start);
"""

print("Building base features (SQL)…")
with engine.begin() as con:
    con.execute(text(features_sql))
print("✅ Base features created in public.features")


regex_iso = r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$"
limit_clause = f"LIMIT {int(ROW_LIMIT)}" if ROW_LIMIT is not None else ""

comments_sql = f"""
WITH oc AS (
  SELECT
    o.user_id,
    o.order_id,
    CASE
      WHEN NULLIF(o.order_date,'') ~ '{regex_iso}'
      THEN (o.order_date)::timestamp ELSE NULL END AS order_ts,
    c.description
  FROM {SCHEMA}.comments c
  JOIN {SCHEMA}.orders   o USING(order_id)
  WHERE coalesce(c.description,'') <> ''
),
cm AS (
  SELECT
    user_id,
    date_trunc('month', order_ts)::date AS month_start,
    description
  FROM oc
  WHERE order_ts IS NOT NULL
)
SELECT user_id, month_start, description
FROM cm
{limit_clause};
"""

print("Fetching comments for sentiment…")
comments = pd.read_sql(comments_sql, engine, parse_dates=["month_start"])
print(f"Comments pulled: {len(comments):,}")
if comments.empty:
    print("No comments with valid timestamps found. (Features table is still built.)")
    sys.exit(0)


try:
    from hazm import Normalizer
    normalizer = Normalizer()
    def fa_norm(s: str) -> str:
        try:
            return normalizer.normalize((s or "").strip())
        except Exception:
            return (s or "").strip()
except Exception:
    print("⚠️  hazm not available; continuing without normalization.")
    def fa_norm(s: str) -> str:
        return (s or "").strip()

comments["text_norm"] = comments["description"].astype(str).map(fa_norm)

try:
    import torch
except Exception as e:
    raise SystemExit(
        "❌ PyTorch is required for Transformers.\n"
        "Install and rerun:\n"
        "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
        f"Original error: {e}"
    )

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
except Exception as e:
    raise SystemExit(
        "❌ Transformers is required.\n"
        "Install and rerun:\n"
        "  pip install 'transformers>=4.35'\n"
        f"Original error: {e}"
    )

print(f"Loading HF model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device_desc = "cpu"
use_pipeline_device = -1

if torch.cuda.is_available():
    model.to("cuda")
    device_desc = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    model.to("mps")
    device_desc = "mps"
else:
    model.to("cpu")
    device_desc = "cpu"

print(f"Using device: {device_desc}")

pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    truncation=True,
    max_length=MAX_LEN,
    top_k=None,
    function_to_apply="softmax",
    device=use_pipeline_device,
)

def detect_label_map(pipeline: TextClassificationPipeline):
    try:
        good = pipeline(["خیلی خوب"])[0]
        bad  = pipeline(["خیلی بد"])[0]
        good_label = max(good, key=lambda d: d["score"])["label"]
        bad_label  = max(bad,  key=lambda d: d["score"])["label"]
        all_labels = {d["label"] for d in good}
        neu_label = None
        if len(all_labels) >= 3:
            neu_label = list(all_labels - {good_label, bad_label})[0]
        return {"pos": good_label, "neg": bad_label, "neu": neu_label}
    except Exception as e:
        print(f"⚠️ Could not auto-detect labels, defaulting to LABEL_1=pos, LABEL_0=neg. Detail: {e}")
        return {"pos": "LABEL_1", "neg": "LABEL_0", "neu": None}

label_map = detect_label_map(pipe)
print("Detected label map:", label_map)


def batched_index(idx_list, n=BATCH_SIZE):
    for i in range(0, len(idx_list), n):
        yield idx_list[i:i+n]

pos_scores, neg_scores, neu_scores = [], [], []
print("Scoring comments (batched)…")
for idxs in tqdm(list(batched_index(comments.index.tolist(), BATCH_SIZE))):
    texts = comments.loc[idxs, "text_norm"].tolist()
    outputs = pipe(texts)
    for outs in outputs:
        scores = {d["label"]: d["score"] for d in outs}
        pos = float(scores.get(label_map["pos"], 0.0))
        neg = float(scores.get(label_map["neg"], 0.0))
        neu = float(scores.get(label_map["neu"], np.nan)) if label_map["neu"] else np.nan
        pos_scores.append(pos); neg_scores.append(neg); neu_scores.append(neu)

comments["fa_pos"] = pos_scores
comments["fa_neg"] = neg_scores
comments["fa_neu"] = neu_scores


sent_month = (comments
              .groupby(["user_id", "month_start"], as_index=False)
              .agg(
                  comments_cnt_with_sent_m=("text_norm", "size"),
                  fa_sent_pos_avg_m=("fa_pos", "mean"),
                  fa_sent_neg_avg_m=("fa_neg", "mean"),
                  fa_sent_neu_avg_m=("fa_neu", "mean"),
              ))
sent_month["fa_sent_compound_avg_m"] = sent_month["fa_sent_pos_avg_m"] - sent_month["fa_sent_neg_avg_m"]

print("Sentiment aggregates (head):")
print(sent_month.head())


with engine.begin() as con:
    sent_month.to_sql("comment_sentiment_month_fa_tmp", con, schema=SCHEMA, if_exists="replace", index=False)
    
    con.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.comment_sentiment_month_fa (
            user_id TEXT NOT NULL,
            month_start DATE NOT NULL,
            comments_cnt_with_sent_m INTEGER,
            fa_sent_pos_avg_m DOUBLE PRECISION,
            fa_sent_neg_avg_m DOUBLE PRECISION,
            fa_sent_neu_avg_m DOUBLE PRECISION,
            fa_sent_compound_avg_m DOUBLE PRECISION,
            PRIMARY KEY (user_id, month_start)
        );
    """))
    con.execute(text(f"""
        INSERT INTO {SCHEMA}.comment_sentiment_month_fa
            (user_id, month_start, comments_cnt_with_sent_m, fa_sent_pos_avg_m,
             fa_sent_neg_avg_m, fa_sent_neu_avg_m, fa_sent_compound_avg_m)
        SELECT user_id, month_start, comments_cnt_with_sent_m, fa_sent_pos_avg_m,
               fa_sent_neg_avg_m, fa_sent_neu_avg_m, fa_sent_compound_avg_m
        FROM {SCHEMA}.comment_sentiment_month_fa_tmp
        ON CONFLICT (user_id, month_start) DO UPDATE SET
            comments_cnt_with_sent_m = EXCLUDED.comments_cnt_with_sent_m,
            fa_sent_pos_avg_m        = EXCLUDED.fa_sent_pos_avg_m,
            fa_sent_neg_avg_m        = EXCLUDED.fa_sent_neg_avg_m,
            fa_sent_neu_avg_m        = EXCLUDED.fa_sent_neu_avg_m,
            fa_sent_compound_avg_m   = EXCLUDED.fa_sent_compound_avg_m;
    """))
    con.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.comment_sentiment_month_fa_tmp;"))

    con.execute(text(f"ALTER TABLE {SCHEMA}.features ADD COLUMN IF NOT EXISTS comments_cnt_with_sent_m INTEGER;"))
    con.execute(text(f"ALTER TABLE {SCHEMA}.features ADD COLUMN IF NOT EXISTS fa_sent_pos_avg_m DOUBLE PRECISION;"))
    con.execute(text(f"ALTER TABLE {SCHEMA}.features ADD COLUMN IF NOT EXISTS fa_sent_neg_avg_m DOUBLE PRECISION;"))
    con.execute(text(f"ALTER TABLE {SCHEMA}.features ADD COLUMN IF NOT EXISTS fa_sent_neu_avg_m DOUBLE PRECISION;"))
    con.execute(text(f"ALTER TABLE {SCHEMA}.features ADD COLUMN IF NOT EXISTS fa_sent_compound_avg_m DOUBLE PRECISION;"))

    con.execute(text(f"""
        UPDATE {SCHEMA}.features f
        SET
          comments_cnt_with_sent_m = s.comments_cnt_with_sent_m,
          fa_sent_pos_avg_m        = s.fa_sent_pos_avg_m,
          fa_sent_neg_avg_m        = s.fa_sent_neg_avg_m,
          fa_sent_neu_avg_m        = s.fa_sent_neu_avg_m,
          fa_sent_compound_avg_m   = s.fa_sent_compound_avg_m
        FROM {SCHEMA}.comment_sentiment_month_fa s
        WHERE f.user_id = s.user_id
          AND f.month_start = s.month_start;
    """))

print("✅ All done. public.features now includes Persian sentiment columns.")
