import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction System",
    layout="wide"
)

# =========================
# LOAD MODEL SAFELY
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

try:
    model, model_columns = load_model()
except Exception:
    st.error("❌ Model could not be loaded. Ensure churn_model.pkl and model_columns.pkl exist.")
    st.stop()

# =========================
# HEADER
# =========================
st.title("Customer Churn Prediction System")
st.write("AI-powered customer retention analytics")

st.divider()

# =========================
# SIDEBAR INPUTS
# =========================
with st.sidebar:
    st.header("Customer Profile")

    age = st.number_input("Age", 18, 100, 30)
    total_orders = st.number_input("Total Orders", 0, 1000, 10)
    total_spend = st.number_input("Total Spend", 0.0, 100000.0, 500.0)
    avg_order_value = st.number_input("Average Order Value", 0.0, 10000.0, 50.0)
    days_since_last_purchase = st.number_input("Days Since Last Purchase", 0, 1000, 30)

    reviews_given = st.number_input("Reviews Given", 0, 500, 5)
    avg_review_score = st.slider("Average Review Score", 0.0, 5.0, 3.5)

    returns_made = st.number_input("Returns Made", 0, 100, 0)
    wishlist_items = st.number_input("Wishlist Items", 0, 500, 10)

    newsletter_subscribed = st.radio("Newsletter Subscribed", ["Yes", "No"])
    newsletter_value = 1 if newsletter_subscribed == "Yes" else 0

# =========================
# PREDICTION
# =========================
if st.button("Predict Churn Risk"):

    input_data = pd.DataFrame([{
        "age": age,
        "total_orders": total_orders,
        "total_spend_usd": total_spend,
        "avg_order_value_usd": avg_order_value,
        "days_since_last_purchase": days_since_last_purchase,
        "reviews_given": reviews_given,
        "avg_review_score": avg_review_score,
        "returns_made": returns_made,
        "wishlist_items": wishlist_items,
        "newsletter_subscribed": newsletter_value
    }])

    # Align columns
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_data)[0]

    # Safe probability handling
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
    else:
        probability = 0.5  # fallback

    # Risk level
    if probability >= 0.7:
        risk = "HIGH RISK"
        color = "red"
    elif probability >= 0.4:
        risk = "MEDIUM RISK"
        color = "orange"
    else:
        risk = "LOW RISK"
        color = "green"

    # =========================
    # RESULTS
    # =========================
    st.subheader("Results")

    st.markdown(f"### Prediction: **{risk}**")
    st.markdown(f"### Churn Probability: **{probability:.2%}**")

    # Chart
    fig, ax = plt.subplots()
    ax.barh(["Churn"], [probability], color=color)
    ax.set_xlim(0, 1)
    ax.set_title("Churn Probability")
    st.pyplot(fig)

    # Insights
    st.subheader("Insights")

    if probability > 0.7:
        st.warning("Customer is likely to churn. Immediate action required.")
    elif probability > 0.4:
        st.info("Customer is at risk. Engage with marketing campaigns.")
    else:
        st.success("Customer is healthy and likely to stay.")