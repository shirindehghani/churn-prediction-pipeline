# 🧠 Customer Churn Prediction Pipeline

A complete end-to-end **data science and engineering pipeline** to predict customer churn based on order, CRM, and comment data.  
This project includes data ingestion, transformation, feature engineering, model training, and API deployment using **FastAPI** and **Docker Compose**.

---

## 📁 Project Structure
```
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


---

## 🧩 Problem Description

You are provided with **orders**, **CRM**, and **comments** data for users over the past months.

The goal is to **predict whether a user will churn next month** based on historical activity, customer feedback, and behavior patterns.

---

## 📊 Data Overview

### 1. Orders
| Column | Description |
|--------|--------------|
| `order_id` | Unique order identifier |
| `user_id` | Unique user identifier |
| `is_otd` | Delivered on time (boolean) |
| `order_date` | Date of order |
| `delivery_status` | Delivery status label |

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

## 🗄️ Database Setup

### Step 1: Load Raw Data to Postgres

Run the following script to create the database schema and load the raw CSVs:

```
python load_to_postgres.py

This script will:
- Connect to the Postgres database
- Create relational tables for orders, crm, and comments
- Insert all raw records into their respective tables

---

### Step 2: Generate Sentiment and other Features
After data is loaded, generate text sentiment and other derived features:
```
python sentiment_features_to_postgres.py

This script:
- Extracts textual sentiment features (e.g., polarity, subjectivity)
- Saves these features into a new database table linked by order_id

---

### Step 3: Create Final Feature Set

Finally, prepare the dataset for model training:
```
python final_features.py

This script:
- Merges CRM, order, and sentiment features
- Aggregates data at the user level
- Generates the final labeled dataset for churn prediction

---

## 📈 Modeling & Evaluation

Explore and train models using the provided notebooks:
```
EDA_Feature_Engineering.ipynb

- Perform exploratory data analysis
- Engineer temporal, behavioral, and satisfaction-based features
- Define churn (e.g., no order in next month)

```
Train_Models.ipynb

- Train and evaluate churn prediction models (e.g., Logistic Regression, XGBoost)
- Use evaluation metrics such as AUC, Precision/Recall, and F1-score
- Interpret model results and identify churn drivers

---

A RESTful API is built using FastAPI in `app/main.py`.

Endpoint

```
POST /predict

Input:
```
{
  "user_id": 12345
}
```
Output:
```
{
  "user_id": 12345,
  "will_churn": "yes",
  "churn_probability": 0.82
}

---

## 🐳 Docker Deployment

The project includes a complete Docker setup for local development.

1. Build and Run Containers
```
docker-compose up --build

---

2. Test the API

Visit http://localhost:8000/docs
 for the interactive Swagger UI.

---

## ⚙️ Requirements

Install all dependencies locally (if not using Docker):
```
pip install -r requirements.txt

```