```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# =========================
# LOAD MODEL SAFELY
# =========================
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns

model, model_columns = load_model()

# =========================
# TITLE
# =========================
st.title("🎯 Customer Churn Prediction System")
st.write("Predict customer churn risk using machine learning")

# =========================
# INPUTS
# =========================
with st.sidebar:
    st.header("Customer Information")

    age = st.number_input("Age", 18, 100, 30)
    total_orders = st.number_input("Total Orders", 0, 1000, 10)
    total_spend = st.number_input("Total Spend", 0.0, 100000.0, 500.0)
    avg_order_value = st.number_input("Avg Order Value", 0.0, 10000.0, 50.0)
    days_since_last = st.number_input("Days Since Last Purchase", 0, 1000, 30)
    reviews = st.number_input("Reviews Given", 0, 500, 5)
    rating = st.slider("Avg Rating", 0.0, 5.0, 3.5)
    returns = st.number_input("Returns Made", 0, 100, 0)
    wishlist = st.number_input("Wishlist Items", 0, 500, 10)
    newsletter = st.radio("Newsletter Subscribed", ["Yes", "No"])
    newsletter_val = 1 if newsletter == "Yes" else 0

# =========================
# PREDICTION
# =========================
def predict():
    data = pd.DataFrame([{
        "age": age,
        "total_orders": total_orders,
        "total_spend_usd": total_spend,
        "avg_order_value_usd": avg_order_value,
        "days_since_last_purchase": days_since_last,
        "reviews_given": reviews,
        "avg_review_score": rating,
        "returns_made": returns,
        "wishlist_items": wishlist,
        "newsletter_subscribed": newsletter_val
    }])

    data = data.reindex(columns=model_columns, fill_value=0)

    prob = model.predict_proba(data)[0][1]
    pred = model.predict(data)[0]

    return pred, prob

# =========================
# RUN PREDICTION
# =========================
if st.button("Predict Churn Risk"):

    pred, prob = predict()

    if prob >= 0.7:
        status = "High Risk"
        color = "red"
    elif prob >= 0.4:
        status = "Medium Risk"
        color = "orange"
    else:
        status = "Low Risk"
        color = "green"

    # =========================
    # RESULTS
    # =========================
    st.subheader("Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Risk Level", status)
    col2.metric("Churn Probability", f"{prob:.1%}")
    col3.metric("Prediction", "Churn" if pred == 1 else "Stay")

    # =========================
    # CHART
    # =========================
    fig, ax = plt.subplots()

    ax.barh(["Churn Risk"], [prob], color=color)
    ax.set_xlim(0, 1)
    ax.axvline(0.4, linestyle="--", color="orange")
    ax.axvline(0.7, linestyle="--", color="red")

    st.pyplot(fig)

    # =========================
    # SUMMARY
    # =========================
    st.subheader("Insight")

    if prob >= 0.7:
        st.error("High churn risk — immediate action needed.")
    elif prob >= 0.4:
        st.warning("Medium churn risk — engagement recommended.")
    else:
        st.success("Low churn risk — customer is stable.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Machine Learning Churn Prediction App")
```
