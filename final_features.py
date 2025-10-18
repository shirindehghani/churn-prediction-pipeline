from __future__ import annotations
from sqlalchemy import create_engine, text

# ---------- DB CONFIG ----------
DB_USER = "sn_dehghani"
DB_PASS = "sndi"
DB_HOST = "localhost"
DB_PORT = "8000"   # change to 5432 if that's your Postgres port
DB_NAME = "sn_dehghani"
SCHEMA  = "public"

# If you prefer month-bucket matching (any timestamp inside the month),
# set MATCH_BY_MONTH = True. Otherwise we'll cast both sides to DATE.
MATCH_BY_MONTH = False

SENTIMENT_COLS = {
    "comments_cnt_with_sent_m",
    "fa_sent_pos_avg_m",
    "fa_sent_neg_avg_m",
    "fa_sent_neu_avg_m",
    "fa_sent_compound_avg_m",
}

def get_features_columns(engine):
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = :schema AND table_name = 'features'
    ORDER BY ordinal_position;
    """
    with engine.begin() as con:
        rows = con.execute(text(sql), {"schema": SCHEMA}).fetchall()
    return [r[0] for r in rows]

def _join_condition():
    """Return the ON clause that robustly matches month_start."""
    if MATCH_BY_MONTH:
        # Align by month bucket
        return (
            "s.user_id = f.user_id "
            "AND date_trunc('month', s.month_start) = date_trunc('month', f.month_start)"
        )
    # Align by exact date (casts remove time/tz parts if present)
    return "s.user_id = f.user_id AND s.month_start::date = f.month_start::date"

def _select_list_with_coalesce(base_cols):
    return ",\n    ".join(
        [f'f."{c}"' for c in base_cols] + [
            "COALESCE(s.comments_cnt_with_sent_m, 0)        AS comments_cnt_with_sent_m",
            "COALESCE(s.fa_sent_pos_avg_m, 0.0)             AS fa_sent_pos_avg_m",
            "COALESCE(s.fa_sent_neg_avg_m, 0.0)             AS fa_sent_neg_avg_m",
            "COALESCE(s.fa_sent_neu_avg_m, 0.0)             AS fa_sent_neu_avg_m",
            "COALESCE(s.fa_sent_compound_avg_m, 0.0)        AS fa_sent_compound_avg_m",
        ]
    )

def _diagnostics(engine):
    """Print a few useful diagnostics before (re)building the table."""
    print("🧪 Running diagnostics …")
    with engine.begin() as con:
        # 1) Do we have any non-NULL sentiment values available?
        diag1 = con.execute(text(f"""
            SELECT
              COUNT(*)                                  AS rows_in_s,
              COUNT(comments_cnt_with_sent_m)           AS cnt_comments_nonnull,
              COUNT(fa_sent_pos_avg_m)                  AS cnt_pos_nonnull,
              COUNT(fa_sent_neg_avg_m)                  AS cnt_neg_nonnull,
              COUNT(fa_sent_neu_avg_m)                  AS cnt_neu_nonnull,
              COUNT(fa_sent_compound_avg_m)             AS cnt_comp_nonnull
            FROM {SCHEMA}.comment_sentiment_month_fa
        """)).mappings().one()
        print(f"   • sentiment table rows: {diag1['rows_in_s']}")
        print(f"   • non-NULL counts — comments:{diag1['cnt_comments_nonnull']}, "
              f"pos:{diag1['cnt_pos_nonnull']}, neg:{diag1['cnt_neg_nonnull']}, "
              f"neu:{diag1['cnt_neu_nonnull']}, comp:{diag1['cnt_comp_nonnull']}")

        # 2) How many rows will match under our join condition?
        join_on = _join_condition()
        diag2 = con.execute(text(f"""
            SELECT
              COUNT(*) AS total_f,
              COUNT(s.user_id) AS matched_rows
            FROM {SCHEMA}.features f
            LEFT JOIN {SCHEMA}.comment_sentiment_month_fa s
              ON {join_on}
        """)).mappings().one()
        print(f"   • features rows: {diag2['total_f']}, rows with match: {diag2['matched_rows']}")

def rebuild_final_features(engine):
    # 1) Inspect columns and exclude any that collide with sentiment columns
    feat_cols = get_features_columns(engine)
    if "user_id" not in feat_cols or "month_start" not in feat_cols:
        raise SystemExit("❌ public.features must contain 'user_id' and 'month_start' columns.")

    base_cols = [c for c in feat_cols if c not in SENTIMENT_COLS]

    # 2) Optional diagnostics to confirm matches/non-NULLs
    _diagnostics(engine)

    # 3) Build SQL
    select_list = _select_list_with_coalesce(base_cols)
    join_on = _join_condition()

    create_sql = f"""
    DROP TABLE IF EXISTS {SCHEMA}.final_features;

    CREATE TABLE {SCHEMA}.final_features AS
    SELECT
        {select_list}
    FROM {SCHEMA}.features f
    LEFT JOIN {SCHEMA}.comment_sentiment_month_fa s
      ON {join_on};
    """

    # 4) Execute DROP + CREATE
    print("🔄 Rebuilding public.final_features …")
    with engine.begin() as con:
        con.execute(text(create_sql))
    print("✅ final_features created")

    # 5) Try to add PK; if duplicates exist, create a non-unique index instead
    try:
        with engine.begin() as con:
            con.execute(text(f"""
                ALTER TABLE {SCHEMA}.final_features
                ADD PRIMARY KEY (user_id, month_start);
            """))
        print("🔐 Primary key added on (user_id, month_start).")
    except Exception as e:
        print(f"⚠️  Could not add primary key: {e}")
        print("ℹ️  Creating a non-unique index instead.")
        with engine.begin() as con:
            con.execute(text(f"""
                CREATE INDEX IF NOT EXISTS ix_final_features_user_month
                ON {SCHEMA}.final_features (user_id, month_start);
            """))
        print("✅ Non-unique index created on (user_id, month_start).")

def main():
    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        future=True
    )
    rebuild_final_features(engine)

if __name__ == "__main__":
    main()
