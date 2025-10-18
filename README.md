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

