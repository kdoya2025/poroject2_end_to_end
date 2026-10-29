# 🎬 Netflix Customer Churn Prediction & Web App Deployment

A full-stack Machine Learning project designed to analyze customer engagement patterns, predict subscriber churn risk, and deliver real-time predictions through a custom Flask web interface.

---

## 📌 Project Overview
Customer churn is one of the most critical metrics for subscription-based business models like Netflix. The goal of this project is to construct an end-to-end data pipeline that extracts insights from user engagement data, trains competitive classification models, and provides actionable churn predictions via a web application.

---

## 🛠️ Tech Stack & Dependencies
* **Core Language:** Python
* **Data Processing & EDA:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost
* **Model Serialization:** Pickle
* **Web Framework:** Flask
* **Frontend:** HTML

---

## 🔄 Project Workflow

### 1. Data Cleaning & Exploratory Data Analysis (EDA)
* Cleaned missing values, checked distributions, and encoded categorical variables.
* Analyzed subscriber usage trends across subscription tiers, regions, and device types.
* Evaluated customer recency (*last login day*) and engagement intensity (*watch hours*) against churn outcome.

### 2. Feature Selection & Preprocessing
Selected high-signal features critical for prediction:
`Age`, `Subscription Type`, `Monthly Fee`, `Watch Hours`, `Last Login Day`, `Region`, `Device`.

### 3. Model Training & Benchmarking
Evaluated five candidate classification models using cross-validation and standard classification metrics (Precision, Recall, F1-Score, ROC-AUC):
* **Logistic Regression**
* **Support Vector Machine (SVM)**
* **Decision Tree Classifier**
* **XGBoost Classifier**
* **Random Forest Classifier** *(Selected model due to optimal balance between precision and recall)*

### 4. Deployment & Web Application
* Serialized the pre-trained Random Forest pipeline and input transformers via `pickle`.
* Built a lightweight **Flask** web application allowing users/managers to input customer attributes and obtain instant churn predictions.

---
