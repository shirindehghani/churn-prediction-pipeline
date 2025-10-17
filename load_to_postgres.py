import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from tqdm import tqdm

DB_USER = "sn_dehghani"
DB_PASS = "sndi"
DB_HOST = "localhost"
DB_PORT = "8000"
DB_NAME = "sn_dehghani"
SCHEMA  = "public"

orders_csv   = "./data/orders.csv"
crm_csv      = "./data/crm.csv"
comments_csv = "./data/order_comments.csv"

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def read_csv_safely(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")

def normalize_cols(df: pd.DataFrame):
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
    )
    return df

orders_raw   = read_csv_safely(orders_csv)
crm_raw      = read_csv_safely(crm_csv)
comments_raw = read_csv_safely(comments_csv)

orders_raw   = normalize_cols(orders_raw)
crm_raw      = normalize_cols(crm_raw)
comments_raw = normalize_cols(comments_raw)

orders_map_candidates = {
    "order_id": "order_id",
    "id": "order_id",
    "orderid": "order_id",

    "user_id": "user_id",
    "userid": "user_id",
    "user": "user_id",

    "is_otd": "is_otd",
    "on_time_delivery": "is_otd",
    "delivered_on_time": "is_otd",
    "is_on_time": "is_otd",

    "order_date": "order_date",
    "order_datetime": "order_date",
    "order_time": "order_date",
    "created_at": "order_date",

    "delivery_status": "delivery_status",
    "status": "delivery_status",
    "delivery_state": "delivery_status",
}

crm_map_candidates = {
    "order_id": "order_id",
    "id": "order_id",
    "orderid": "order_id",

    "crm_delivery_request_count": "crm_delivery_request_count",
    "delivery_request_count": "crm_delivery_request_count",
    "tickets_delivery_request": "crm_delivery_request_count",

    "crm_fake_delivery_request_count": "crm_fake_delivery_request_count",
    "fake_delivery_request_count": "crm_fake_delivery_request_count",

    "customer_rate": "customer_rate",
    "customer_rating": "customer_rate",
    "shop_rating": "customer_rate",

    "courier_rate": "courier_rate",
    "courier_rating": "courier_rate",
    "driver_rating": "courier_rate",
}

comments_map_candidates = {
    "order_id": "order_id",
    "id": "order_id",
    "orderid": "order_id",

    "description": "description",
    "comment": "description",
    "comments": "description",
    "text": "description",
    "body": "description",
    "message": "description",
}

def remap_and_select(df: pd.DataFrame, mapping: dict, required: list) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        if col in mapping:
            rename[col] = mapping[col]
    df = df.rename(columns=rename)
    present = [c for c in required if c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        for m in missing:
            df[m] = np.nan
        present = required
    return df[present]

orders = remap_and_select(
    orders_raw, orders_map_candidates,
    ["order_id", "user_id", "is_otd", "order_date", "delivery_status"]
)
crm = remap_and_select(
    crm_raw, crm_map_candidates,
    ["order_id", "crm_delivery_request_count", "crm_fake_delivery_request_count", "customer_rate", "courier_rate"]
)
comments = remap_and_select(
    comments_raw, comments_map_candidates,
    ["order_id", "description"]
)


orders["order_id"] = pd.to_numeric
orders["user_id"] = pd.to_numeric(orders["user_id"], errors="coerce")

orders = orders.dropna(subset=["order_id", "user_id"])
orders["order_id"] = orders["order_id"].astype("int64")
orders["user_id"] = orders["user_id"].astype("int64")

valid_order_ids = set(orders["order_id"].unique())
if "order_id" in crm.columns:
    crm = crm[pd.to_numeric(crm["order_id"], errors="coerce").isin(valid_order_ids)]
    crm["order_id"] = crm["order_id"].astype("int64")
if "order_id" in comments.columns:
    comments = comments[pd.to_numeric(comments["order_id"], errors="coerce").isin(valid_order_ids)]
    comments["order_id"] = comments["order_id"].astype("int64")


orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")


for col in ["customer_rate", "courier_rate", "crm_delivery_request_count", "crm_fake_delivery_request_count"]:
    if col in crm.columns:
        crm[col] = pd.to_numeric(crm[col], errors="coerce")

orders = orders.drop_duplicates(subset=["order_id"])
crm = crm.drop_duplicates(subset=["order_id"])
comments = comments.drop_duplicates(subset=["order_id"])

print(f"Normalized shapes -> orders: {orders.shape}, crm: {crm.shape}, comments: {comments.shape}")

# ------------------- DDL & TRUNCATE -------------------
with engine.begin() as conn:
    conn.execute(text(f"""
    CREATE SCHEMA IF NOT EXISTS {SCHEMA};

    CREATE TABLE IF NOT EXISTS {SCHEMA}.orders (
      order_id BIGINT PRIMARY KEY,
      user_id BIGINT NOT NULL,
      is_otd BOOLEAN,
      order_date TIMESTAMP,
      delivery_status TEXT
    );

    CREATE TABLE IF NOT EXISTS {SCHEMA}.crm (
      order_id BIGINT PRIMARY KEY,
      crm_delivery_request_count INT,
      crm_fake_delivery_request_count INT,
      customer_rate NUMERIC(4,2),
      courier_rate NUMERIC(4,2),
      CONSTRAINT fk_crm_order FOREIGN KEY(order_id) REFERENCES {SCHEMA}.orders(order_id)
    );

    CREATE TABLE IF NOT EXISTS {SCHEMA}.comments (
      order_id BIGINT PRIMARY KEY,
      description TEXT,
      CONSTRAINT fk_comments_order FOREIGN KEY(order_id) REFERENCES {SCHEMA}.orders(order_id)
    );

    -- Helpful indexes
    CREATE INDEX IF NOT EXISTS idx_orders_user_date ON {SCHEMA}.orders(user_id, order_date);
    CREATE INDEX IF NOT EXISTS idx_orders_date ON {SCHEMA}.orders(order_date);
    """))
    print("✅ Tables ensured / indexes created.")

    # TRUNCATE in dependency-safe order (children → parent)
    conn.execute(text(f"TRUNCATE TABLE {SCHEMA}.crm, {SCHEMA}.comments, {SCHEMA}.orders;"))
    print("🧹 Tables truncated.")

# ------------------- INSERT (parent → children) -------------------
for name, df in tqdm([('orders', orders), ('crm', crm), ('comments', comments)]):
    if df.empty:
        print(f"ℹ️ {name}: empty dataframe, skipping insert.")
        continue
    df.to_sql(name, engine, schema=SCHEMA, if_exists='append', index=False, method='multi', chunksize=100_000)
    print(f"✅ Inserted {name} ({len(df)} rows)")

# ------------------- VERIFY COUNTS -------------------
with engine.begin() as conn:
    o_cnt = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.orders")).scalar()
    c_cnt = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.crm")).scalar()
    m_cnt = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.comments")).scalar()

print(f"🎉 Done. orders={o_cnt:,} | crm={c_cnt:,} | comments={m_cnt:,}")
