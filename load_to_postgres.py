from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import (
    create_engine,
    String,
    Text,
    BigInteger,
    Identity,
    text,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from dotenv import load_dotenv
import os

load_dotenv()


DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
SCHEMA  = "public"

orders_csv   = "./data/orders.csv"
crm_csv      = "./data/crm.csv"
comments_csv = "./data/order_comments.csv"

CHUNK_SIZE = 10_000
ECHO_SQL   = False


class Base(DeclarativeBase):
    pass


class Orders(Base):
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_orders_user_date", "user_id", "order_date"),
        {"schema": SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=False), primary_key=True)
    order_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    is_otd: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    order_date: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    delivery_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class CRM(Base):
    __tablename__ = "crm"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=False), primary_key=True)

    order_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    crm_delivery_request_count: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    crm_fake_delivery_request_count: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    rate_to_shop: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    rate_to_courier: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class Comments(Base):
    __tablename__ = "comments"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=False), primary_key=True)

    order_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)



def _read_csv_verbatim(path: str | Path) -> pd.DataFrame:
    """
    Preserve values exactly:
      - dtype=str            -> everything is a string
      - keep_default_na=False, na_filter=False -> "" stays "", not NaN
    """
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )


def _normalize_orders_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for raw in df.columns:
        key = re.sub(r"[^a-z]", "", raw.lower())
        if key == "orderid":
            rename_map[raw] = "order_id"
        elif key == "userid":
            rename_map[raw] = "user_id"
        elif key in {"isotd", "is_ontime", "ontime"}:
            rename_map[raw] = "is_otd"
        elif key in {"orderdate", "createdat", "timestamp"}:
            rename_map[raw] = "order_date"
        elif key in {"deliverystatus", "status"}:
            rename_map[raw] = "delivery_status"

    df = df.rename(columns=rename_map)

    cols = ["order_id", "user_id", "is_otd", "order_date", "delivery_status"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()


def _normalize_crm_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for raw in df.columns:
        key = re.sub(r"[^a-z]", "", raw.lower())
        if key == "orderid":
            rename_map[raw] = "order_id"
        elif key in {"crmdeliveryrequestcount", "deliveryrequestcount", "deliveryrequests"}:
            rename_map[raw] = "crm_delivery_request_count"
        elif key in {"crmfakedeliveryrequestcount", "fakedeliveryrequestcount", "fakedeliveryrequests"}:
            rename_map[raw] = "crm_fake_delivery_request_count"
        elif key in {"customerrate", "customerrating", "shoprate", "shoprating", "ratecustomer", "ratingcustomer", "userrate", "userrating"}:
            rename_map[raw] = "rate_to_shop"
        elif key in {"courierrate", "courierrating", "driverrate", "driverrating", "riderrate", "riderrating"}:
            rename_map[raw] = "rate_to_courier"

    df = df.rename(columns=rename_map)

    cols = ["order_id", "crm_delivery_request_count", "crm_fake_delivery_request_count", "rate_to_shop", "rate_to_courier"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()


def _normalize_comments_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for raw in df.columns:
        key = re.sub(r"[^a-z]", "", raw.lower())
        if key in {"orderid"}:
            rename_map[raw] = "order_id"
        elif key in {
            "description", "comment", "comments", "text", "content",
            "review", "reviews", "message", "feedback", "note", "notes",
            "commenttext", "comment_body", "commentbody", "body"
        }:
            rename_map[raw] = "description"

    df = df.rename(columns=rename_map)

    cols = ["order_id", "description"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()


def insert_orders(session: Session, df: pd.DataFrame) -> int:
    df = _normalize_orders_columns(df)
    total = 0
    for i in range(0, len(df), CHUNK_SIZE):
        chunk = df.iloc[i:i + CHUNK_SIZE]
        if chunk.empty:
            continue
        payload = chunk.to_dict(orient="records")
        session.execute(Orders.__table__.insert(), payload)
        total += len(chunk)
    return total


def insert_crm(session: Session, df: pd.DataFrame) -> int:
    df = _normalize_crm_columns(df)
    total = 0
    for i in range(0, len(df), CHUNK_SIZE):
        chunk = df.iloc[i:i + CHUNK_SIZE]
        if chunk.empty:
            continue
        payload = chunk.to_dict(orient="records")
        session.execute(CRM.__table__.insert(), payload)
        total += len(chunk)
    return total


def insert_comments(session: Session, df: pd.DataFrame) -> int:
    df = _normalize_comments_columns(df)
    total = 0
    for i in range(0, len(df), CHUNK_SIZE):
        chunk = df.iloc[i:i + CHUNK_SIZE]
        if chunk.empty:
            continue
        payload = chunk.to_dict(orient="records")
        session.execute(Comments.__table__.insert(), payload)
        total += len(chunk)
    return total


def print_table_counts(session: Session):
    for tbl in ["orders", "crm", "comments"]:
        res = session.execute(text(f'SELECT COUNT(*) FROM "{SCHEMA}"."{tbl}"'))
        print(f"📦 {tbl} rows in DB: {res.scalar_one()}")


def main():
    db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print(f"Connecting to: {db_url} (schema={SCHEMA})")

    engine = create_engine(db_url, echo=ECHO_SQL, future=True, pool_pre_ping=True)

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        conn.execute(text(f"SET search_path TO {SCHEMA}"))
        Base.metadata.create_all(conn)

    print("Reading CSVs...")
    orders_df = _read_csv_verbatim(orders_csv)
    crm_df = _read_csv_verbatim(crm_csv)
    comments_df = _read_csv_verbatim(comments_csv)

    with Session(engine) as session:
        session.execute(text(f"SET search_path TO {SCHEMA}"))

        print("Inserting orders...")
        n_orders = insert_orders(session, orders_df)
        session.commit()
        print(f"Orders inserted: {n_orders}")
        print_table_counts(session)

        print("Inserting CRM...")
        n_crm = insert_crm(session, crm_df)
        session.commit()
        print(f"CRM inserted: {n_crm}")
        print_table_counts(session)

        print("Inserting comments...")
        n_comments = insert_comments(session, comments_df)
        session.commit()
        print(f"Comments inserted: {n_comments}")
        print_table_counts(session)

        print("✅ All rows inserted (no skips).")


if __name__ == "__main__":
    main()
