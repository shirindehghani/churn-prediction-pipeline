CREATE SCHEMA IF NOT EXISTS public;

-- Base tables
CREATE TABLE IF NOT EXISTS orders (
  order_id        BIGINT PRIMARY KEY,
  user_id         BIGINT NOT NULL,
  is_otd          BOOLEAN,
  order_date      TIMESTAMP NOT NULL,
  delivery_status TEXT
);

CREATE TABLE IF NOT EXISTS crm (
  order_id                         BIGINT PRIMARY KEY,
  crm_delivery_request_count       INT,
  crm_fake_delivery_request_count  INT,
  customer_rate                    NUMERIC(4,2),
  courier_rate                     NUMERIC(4,2),
  CONSTRAINT fk_crm_order FOREIGN KEY(order_id) REFERENCES orders(order_id)
);

CREATE TABLE IF NOT EXISTS comments (
  order_id     BIGINT PRIMARY KEY,
  description  TEXT,
  CONSTRAINT fk_comments_order FOREIGN KEY(order_id) REFERENCES orders(order_id)
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_orders_user_date ON orders(user_id, order_date);
CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
