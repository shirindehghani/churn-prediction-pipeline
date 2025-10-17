DROP TABLE IF EXISTS user_month_features CASCADE;

WITH o AS (
  SELECT
    user_id,
    order_id,
    order_date::date AS order_date,
    date_trunc('month', order_date)::date AS month_date,
    is_otd::int AS is_otd_int,
    delivery_status
  FROM orders
),
o_enriched AS (
  SELECT
    o.*,
    c.crm_delivery_request_count,
    c.crm_fake_delivery_request_count,
    c.customer_rate,
    c.courier_rate,
    com.description
  FROM o
  LEFT JOIN crm c USING(order_id)
  LEFT JOIN comments com USING(order_id)
),
-- sentiment (placeholder): we’ll compute in Python; here just keep text.
base AS (
  SELECT
    user_id, month_date,
    COUNT(*) AS orders_cnt_m,
    AVG(is_otd_int) AS otd_rate_m,
    AVG(customer_rate) AS avg_customer_rate_m,
    AVG(courier_rate) AS avg_courier_rate_m,
    SUM(COALESCE(crm_delivery_request_count,0)) AS crm_req_cnt_m,
    SUM(COALESCE(crm_fake_delivery_request_count,0)) AS crm_fake_cnt_m,
    MAX(order_date) AS last_order_date_m,
    ARRAY_AGG(description) FILTER (WHERE description IS NOT NULL) AS comments_list_m
  FROM o_enriched
  GROUP BY user_id, month_date
),
roll AS (
  SELECT
    b.*,
    -- 3-month rolling (including current)
    SUM(orders_cnt_m) OVER (PARTITION BY user_id ORDER BY month_date
                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS orders_cnt_3m,
    AVG(otd_rate_m)   OVER (PARTITION BY user_id ORDER BY month_date
                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS otd_rate_3m,
    SUM(crm_req_cnt_m) OVER (PARTITION BY user_id ORDER BY month_date
                             ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS crm_req_cnt_3m,
    SUM(crm_fake_cnt_m) OVER (PARTITION BY user_id ORDER BY month_date
                              ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS crm_fake_cnt_3m
  FROM base b
),
with_recency AS (
  SELECT
    r.*,
    -- days since last order up to month end
    (date_trunc('month', r.month_date) + INTERVAL '1 month' - INTERVAL '1 day')::date AS month_end,
    GREATEST(0, ( (date_trunc('month', r.month_date) + INTERVAL '1 month' - INTERVAL '1 day')::date
                  - r.last_order_date_m ) ) AS recency_days
  FROM roll r
),
calendar AS (
  SELECT DISTINCT date_trunc('month', order_date)::date AS month_date FROM orders
),
labels AS (
  SELECT
    u.user_id,
    u.month_date,
    CASE WHEN nx.user_id IS NULL THEN 1 ELSE 0 END AS label_churn_next_month
  FROM (
    SELECT DISTINCT user_id, month_date FROM with_recency
  ) u
  LEFT JOIN (
    SELECT user_id, month_date FROM base
  ) nx
  ON nx.user_id = u.user_id
  AND nx.month_date = (u.month_date + INTERVAL '1 month')::date
)
SELECT
  w.user_id, w.month_date,
  orders_cnt_m, orders_cnt_3m,
  otd_rate_m, otd_rate_3m,
  avg_customer_rate_m, avg_courier_rate_m,
  crm_req_cnt_m, crm_req_cnt_3m,
  crm_fake_cnt_m, crm_fake_cnt_3m,
  recency_days,
  comments_list_m,
  l.label_churn_next_month AS label
INTO user_month_features
FROM with_recency w
JOIN labels l USING(user_id, month_date);

CREATE INDEX IF NOT EXISTS idx_umf_user_month ON user_month_features(user_id, month_date);
