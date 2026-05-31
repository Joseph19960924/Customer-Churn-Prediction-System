# 🛒 Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether an e-commerce customer is likely to churn based on behavioral, transactional, and engagement data.

This project includes data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit.

---

## 📌 Problem Statement

Customer churn is a major challenge in e-commerce businesses. Retaining existing customers is more cost-effective than acquiring new ones.

This project aims to predict whether a customer will churn so that businesses can take proactive retention actions.

---

## 📊 Dataset Overview

The dataset contains customer behavioral and transactional data including:

- Age
- Total orders
- Total spend
- Average order value
- Days since last purchase
- Reviews and ratings
- Returns made
- Wishlist activity
- Newsletter subscription status

---

## ⚙️ Technologies Used

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- Streamlit (for deployment)
- Joblib / Pickle
- Matplotlib / Seaborn (EDA)

---

## 🧠 Machine Learning Approach

### 1. Data Preprocessing
- Handled missing values
- Encoded categorical variables
- Feature selection

### 2. Model Training
- Random Forest Classifier
- Train-test split (80/20)

### 3. Class Imbalance Handling
- Used class weighting to improve churn detection

### 4. Evaluation Metrics
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 📈 Model Performance

- Accuracy: ~82%
- Churn Recall: ~0.54
- Churn Precision: ~0.30

> Focus was placed on improving recall to better identify at-risk customers.

---

## 🚀 Deployment

The model is deployed using Streamlit as an interactive web application.

### Run locally:

```bash
streamlit run app.py


customer-churn-project/
│
├── app.py
├── churn_model.pkl
├── model_columns.pkl
├── requirements.txt
└── README.md