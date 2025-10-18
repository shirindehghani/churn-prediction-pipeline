<!-- ========================================================= -->
<!-- 🧠 PROJECT TITLE AND INTRODUCTION -->
<!-- ========================================================= -->

# 🧠 Customer Churn Prediction Pipeline
<!-- `#` creates the main project title -->

A complete end-to-end **data science and engineering pipeline** to predict customer churn based on order, CRM, and comment data.  
<!-- `**bold**` emphasizes keywords -->
This project includes data ingestion, transformation, feature engineering, model training, and API deployment using **FastAPI** and **Docker Compose**.

Note that you should create `.env` file like `sample.env`

---

<!-- ========================================================= -->
<!-- 📁 PROJECT STRUCTURE -->
<!-- ========================================================= -->

## 📁 Project Structure
<!-- `##` marks a second-level heading -->

---

```bash
CHURN-PIPELINE/
│
├── app/
│ ├── init.py
│ ├── main.py # FastAPI application
│
├── data/
│ ├── crm.csv # CRM dataset
│ ├── order_comments.csv # Customer comments dataset
│ ├── orders.csv # Orders dataset
│
├── notebooks/
│ ├── EDA_Feature_Engineering.ipynb # Exploratory Data Analysis & Feature Engineering
│ ├── Train_Models.ipynb # Model training & evaluation
│
├── venv/ # Virtual environment (ignored in Docker)
│
├── load_to_postgres.py # Load & transform raw CSVs into Postgres database
├── sentiment_features_to_postgres.py # Generate sentiment-based features from comments
├── final_features.py # Combine all features into a final training dataset
│
├── requirements.txt # Python dependencies
├── docker-compose.yml # Compose file for local setup
├── Dockerfile # FastAPI service Docker image
├── .dockerignore
├── .gitignore
```
<!-- Triple backticks with no language specified render monospaced directory trees -->

---

<!-- ========================================================= -->
<!-- 🧩 PROBLEM DESCRIPTION -->
<!-- ========================================================= -->

## 🧩 Problem Description

You are provided with **orders**, **CRM**, and **comments** data for users over the past months.  

The goal is to **predict whether a user will churn next month** based on historical activity, customer feedback, and behavior patterns.

---

<!-- ========================================================= -->
<!-- 📊 DATA OVERVIEW -->
<!-- ========================================================= -->

## 📊 Data Overview

### 1. Orders
<!-- `###` creates a third-level heading -->

| Column | Description |
|--------|--------------|
| `order_id` | Unique order identifier |
| `user_id` | Unique user identifier |
| `is_otd` | Delivered on time (boolean) |
| `order_date` | Date of order |
| `delivery_status` | Delivery status label |
<!-- Markdown tables use pipes `|` and dashes `-` -->

### 2. CRM

| Column | Description |
|--------|--------------|
| `order_id` | Unique order identifier |
| `crm_delivery_request_count` | Count of delivery request tickets |
| `crm_fake_delivery_request_count` | Count of fake delivery requests |
| `customer_rate` | Rating given by the customer for the shop |
| `courier_rate` | Rating given for the courier |

### 3. Comments

| Column | Description |
|--------|--------------|
| `order_id` | Unique order identifier |
| `description` | Free-text comment provided by the customer |

---

<!-- ========================================================= -->
<!-- 🗄️ DATABASE SETUP -->
<!-- ========================================================= -->

## 🗄️ Database Setup

### Step 1: Load Raw Data to Postgres

Run the following script to create the database schema and load the raw CSVs:

```bash
python load_to_postgres.py
```

This script will:
- Connect to the Postgres database
- Create relational tables for orders, crm, and comments
- Insert all raw records into their respective tables

---

### Step 2: Generate Sentiment and Other Features

After data is loaded, generate text sentiment and other derived features:

```bash
python sentiment_features_to_postgres.py
```

This script:
- Extracts textual sentiment features (e.g., polarity, subjectivity)
- Saves these features into a new database table linked by order_id

---

### Step 3: Create Final Feature Set

Finally, prepare the dataset for model training:

```bash
python final_features.py
```

This script:
- Merges CRM, order, and sentiment features
- Aggregates data at the user level
- Generates the final labeled dataset for churn prediction

---

## 📈 Modeling & Evaluation

Explore and train models using the provided notebooks:
```bash
python EDA_Feature_Engineering.ipynb
```

- Perform exploratory data analysis
- Engineer temporal, behavioral, and satisfaction-based features
- Define churn (e.g., no order in next month)

```bash
python Train_Models.ipynb
```

- Train and evaluate churn prediction models (e.g., Logistic Regression, XGBoost)
- Use evaluation metrics such as AUC, Precision/Recall, and F1-score
- Interpret model results and identify churn drivers

---

## 🚀 API Endpoint
A RESTful API is built using FastAPI in `app/main.py`.

#### Endpoint
```bash
POST /predict
```

---

#### Input
```bash
{
    "user_id": 12345
    }
```

---

### Output
```bash
{
  "user_id": 12345,
  "will_churn": "yes",
  "churn_probability": 0.82
}
```

---

## 🐳 Docker Deployment

The project includes a complete Docker setup for local development.

```bash
docker-compose up --build
```

---

## ⚙️ Requirements

Install all dependencies locally (if not using Docker):

```bash
pip install -r requirements.txt
```


@Author : Shirin Dehghani/ AI Engineer