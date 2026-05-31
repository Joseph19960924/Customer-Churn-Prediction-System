import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .risk-high {
        background: #ffebee;
        border-left: 5px solid #c62828;
        padding: 1rem;
        border-radius: 5px;
    }
    .risk-medium {
        background: #fff3e0;
        border-left: 5px solid #f9a825;
        padding: 1rem;
        border-radius: 5px;
    }
    .risk-low {
        background: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎯 Customer Churn Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">AI-Powered Customer Retention Analytics</p>', unsafe_allow_html=True)
st.divider()

# Sidebar inputs
with st.sidebar:
    st.markdown("## 📊 Customer Profile")
    
    age = st.number_input("Age", 18, 100, 35)
    total_orders = st.number_input("Total Orders", 0, 500, 15)
    total_spend = st.number_input("Total Spend (USD)", 0, 100000, 750)
    avg_order_value = st.number_input("Avg Order Value (USD)", 0, 1000, 50)
    days_since_last = st.number_input("Days Since Last Purchase", 0, 365, 30)
    reviews = st.number_input("Reviews Given", 0, 200, 5)
    rating = st.slider("Avg Review Rating", 0.0, 5.0, 3.5, 0.1)
    returns = st.number_input("Returns Made", 0, 100, 1)
    wishlist = st.number_input("Wishlist Items", 0, 200, 8)
    newsletter = st.radio("Newsletter Subscribed", ["Yes", "No"], horizontal=True)
    newsletter_val = 1 if newsletter == "Yes" else 0
    
    predict_btn = st.button("🎯 Predict Churn Risk", type="primary", use_container_width=True)

# Prediction function
def predict_churn(age, orders, spend, avg_value, days_since, reviews, rating, returns, wishlist, newsletter):
    # Simple rule-based prediction
    probability = 0.0
    
    if days_since > 60:
        probability += 0.3
    if orders < 5:
        probability += 0.2
    if returns > orders * 0.2:
        probability += 0.2
    if rating < 2.5:
        probability += 0.15
    if newsletter == 0:
        probability += 0.05
        
    return min(probability, 0.95)

# Main logic
if predict_btn:
    # Make prediction
    probability = predict_churn(
        age, total_orders, total_spend, avg_order_value, 
        days_since_last, reviews, rating, returns, wishlist, newsletter_val
    )
    
    # Determine risk level
    if probability >= 0.7:
        risk_level = "High Risk"
        risk_class = "risk-high"
        risk_color = "#c62828"
        status = "⚠️ Will Likely Churn"
    elif probability >= 0.4:
        risk_level = "Medium Risk"
        risk_class = "risk-medium"
        risk_color = "#f9a825"
        status = "⚠️ At Risk"
    else:
        risk_level = "Low Risk"
        risk_class = "risk-low"
        risk_color = "#2e7d32"
        status = "✅ Likely to Stay"
    
    # Display results
    st.markdown("## 📊 Analysis Results")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:{risk_color};">{status}</h3>
            <p>Prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{probability:.1%}</h3>
            <p>Churn Probability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:{risk_color};">{risk_level}</h3>
            <p>Risk Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        confidence = abs(probability - 0.5) * 2
        st.markdown(f"""
        <div class="metric-card">
            <h3>{confidence:.1%}</h3>
            <p>Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability gauge
    st.markdown("---")
    st.markdown("## 📈 Risk Analysis")
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(["Churn Risk"], [probability], color=risk_color, height=0.3)
    ax.set_xlim(0, 1)
    ax.axvline(x=0.4, color="orange", linestyle="--", label="Medium Risk (40%)")
    ax.axvline(x=0.7, color="red", linestyle="--", label="High Risk (70%)")
    ax.text(probability, 0, f"{probability:.1%}", ha="center", va="center", fontweight="bold")
    ax.legend(loc="lower right")
    st.pyplot(fig)
    
    # Risk factors
    st.markdown("---")
    st.markdown("## 🎯 Risk Assessment")
    
    risk_factors = []
    good_factors = []
    
    if days_since_last > 60:
        risk_factors.append(f"Long inactivity ({days_since_last} days)")
    if total_orders < 5:
        risk_factors.append(f"Low purchase frequency ({total_orders} orders)")
    if returns > total_orders * 0.2:
        risk_factors.append(f"High return rate ({returns} returns)")
    if rating < 2.5:
        risk_factors.append("Low satisfaction ratings")
    if newsletter_val == 0:
        risk_factors.append("Not subscribed to newsletter")
    
    if days_since_last < 30:
        good_factors.append("Recent purchase activity")
    if total_orders > 20:
        good_factors.append(f"Loyal customer ({total_orders} orders)")
    if rating > 4.0:
        good_factors.append("High satisfaction ratings")
    if newsletter_val == 1:
        good_factors.append("Subscribed to newsletter")
    
    st.markdown(f"""
    <div class="{risk_class}">
        <strong>Risk Level: {risk_level}</strong><br>
        Churn Probability: {probability:.1%}<br><br>
        
        <strong>⚠️ Risk Factors:</strong><br>
        {chr(10).join(['• ' + f for f in risk_factors]) if risk_factors else '✅ No significant risks'}<br><br>
        
        <strong>✅ Positive Factors:</strong><br>
        {chr(10).join(['• ' + f for f in good_factors]) if good_factors else 'ℹ️ No significant positives'}
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("---")
    st.markdown("## 💡 Recommendations")
    
    if probability >= 0.7:
        st.error("""
        **Immediate Actions Required:**
        - Send personalized discount offer (20-30% off)
        - Proactive customer outreach
        - Request feedback survey
        - Invite to loyalty program
        """)
    elif probability >= 0.4:
        st.warning("""
        **Preventive Actions:**
        - Send personalized recommendations
        - Invite to loyalty program
        - Share educational content
        - Offer free shipping
        """)
    else:
        st.success("""
        **Growth Actions:**
        - Upsell and cross-sell opportunities
        - Referral program invitation
        - VIP program consideration
        - Send thank you note
        """)
    
    # Value analysis
    st.markdown("---")
    st.markdown("## 💰 Value Analysis")
    
    clv = total_spend * (1 - probability) * 2
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Value", f"${total_spend:,.0f}")
    with col2:
        st.metric("Est. Lifetime Value", f"${clv:,.0f}")
    with col3:
        retention = (1 - probability) * 100
        st.metric("Retention Potential", f"{retention:.0f}%")

else:
    st.info("👈 Enter customer details in the sidebar and click 'Predict Churn Risk'")
    
    # Show features
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Features\n- Churn scoring\n- Risk factors\n- Confidence metrics")
    with col2:
        st.markdown("### 🎯 Risk Levels\n- 🟢 Low: <40%\n- 🟡 Medium: 40-70%\n- 🔴 High: >70%")
    with col3:
        st.markdown("### 💡 Benefits\n- Reduce churn\n- Increase retention\n- Maximize value")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #999;">
    <p>🤖 Powered by AI | 📊 Analyzes customer behavior | 💡 Actionable insights</p>
</div>
""", unsafe_allow_html=True)