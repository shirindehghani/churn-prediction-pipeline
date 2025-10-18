<!-- ========================================================= -->
<!-- 🧠 PROJECT TITLE AND INTRODUCTION -->
<!-- ========================================================= -->

# 🧠 Customer Churn Prediction Pipeline
<!-- `#` creates the main project title -->

A complete end-to-end **data science and engineering pipeline** to predict customer churn based on order, CRM, and comment data.  
<!-- `**bold**` emphasizes keywords -->
This project includes data ingestion, transformation, feature engineering, model training, and API deployment using **FastAPI** and **Docker Compose**.

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
