import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Try to import optional packages
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CREATE SAMPLE MODEL (if needed)
# =========================
@st.cache_resource
def get_model():
    """Create a simple prediction model"""
    np.random.seed(42)
    
    # Define feature columns
    feature_columns = [
        "age", "total_orders", "total_spend_usd", "avg_order_value_usd",
        "days_since_last_purchase", "reviews_given", "avg_review_score",
        "returns_made", "wishlist_items", "newsletter_subscribed"
    ]
    
    if SKLEARN_AVAILABLE:
        # Create a simple random forest model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Generate synthetic training data
        n_samples = 500
        X_train = pd.DataFrame({
            "age": np.random.randint(18, 70, n_samples),
            "total_orders": np.random.poisson(10, n_samples),
            "total_spend_usd": np.random.exponential(500, n_samples),
            "avg_order_value_usd": np.random.exponential(50, n_samples),
            "days_since_last_purchase": np.random.exponential(30, n_samples),
            "reviews_given": np.random.poisson(3, n_samples),
            "avg_review_score": np.random.uniform(1, 5, n_samples),
            "returns_made": np.random.poisson(1, n_samples),
            "wishlist_items": np.random.poisson(5, n_samples),
            "newsletter_subscribed": np.random.binomial(1, 0.3, n_samples)
        })
        
        # Create churn logic
        y_train = (
            (X_train["days_since_last_purchase"] > 60) * 0.3 +
            (X_train["total_orders"] < 5) * 0.2 +
            (X_train["returns_made"] > X_train["total_orders"] * 0.2) * 0.2 +
            (X_train["avg_review_score"] < 2.5) * 0.15
        ) > 0.4
        
        model.fit(X_train, y_train)
        return model, feature_columns
    else:
        # Return a simple rule-based model
        return None, feature_columns

# Load model
model, feature_columns = get_model()

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px;
    }
    .risk-high {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #c62828;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 5px solid #f9a825;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<p class="main-header">🎯 Customer Churn Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">AI-Powered Customer Retention Analytics</p>', unsafe_allow_html=True)
st.divider()

# =========================
# SIDEBAR INPUTS
# =========================
with st.sidebar:
    st.markdown("## 📊 Customer Profile")
    st.markdown("---")
    
    st.markdown("**👤 Demographics**")
    age = st.number_input("Age", 18, 100, 35, help="Customer age in years")
    
    st.markdown("**🛍️ Purchase Behavior**")
    total_orders = st.number_input("Total Orders", 0, 500, 15, help="Total number of orders")
    total_spend = st.number_input("Total Spend (USD)", 0.0, 100000.0, 750.0, help="Total amount spent")
    avg_order_value = st.number_input("Avg Order Value (USD)", 0.0, 10000.0, 50.0, help="Average order value")
    days_since_last = st.number_input("Days Since Last Purchase", 0, 365, 30, help="Days since last order")
    
    st.markdown("**⭐ Engagement**")
    reviews = st.number_input("Reviews Given", 0, 200, 5, help="Number of reviews written")
    rating = st.slider("Avg Review Rating", 0.0, 5.0, 3.5, 0.1, help="Average rating given")
    returns = st.number_input("Returns Made", 0, 100, 1, help="Number of returns")
    wishlist = st.number_input("Wishlist Items", 0, 200, 8, help="Items in wishlist")
    
    st.markdown("**📧 Marketing**")
    newsletter = st.radio("Newsletter Subscribed", ["Yes", "No"], horizontal=True)
    newsletter_val = 1 if newsletter == "Yes" else 0
    
    st.markdown("---")
    predict_btn = st.button("🎯 Predict Churn Risk", type="primary", use_container_width=True)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_churn(features):
    """Make prediction using the model"""
    if model is not None and SKLEARN_AVAILABLE:
        # Use ML model
        prob = model.predict_proba(features)[0][1]
        pred = model.predict(features)[0]
        return prob, pred
    else:
        # Rule-based prediction
        score = 0
        if days_since_last > 60:
            score += 0.3
        if total_orders < 5:
            score += 0.2
        if returns > total_orders * 0.2:
            score += 0.2
        if rating < 2.5:
            score += 0.15
        if newsletter_val == 0:
            score += 0.05
            
        prob = min(score, 0.95)
        pred = 1 if prob > 0.4 else 0
        return prob, pred

# =========================
# MAIN LOGIC
# =========================
if predict_btn:
    # Prepare input
    input_df = pd.DataFrame([{
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
    
    # Align columns
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    # Predict
    with st.spinner("Analyzing customer behavior..."):
        probability, prediction = predict_churn(input_df)
    
    # Determine risk
    if probability >= 0.7:
        risk = "High Risk"
        risk_class = "risk-high"
        status = "⚠️ Will Likely Churn"
        emoji = "🔴"
        color = "#c62828"
    elif probability >= 0.4:
        risk = "Medium Risk"
        risk_class = "risk-medium"
        status = "⚠️ At Risk"
        emoji = "🟡"
        color = "#f9a825"
    else:
        risk = "Low Risk"
        risk_class = "risk-low"
        status = "✅ Likely to Stay"
        emoji = "🟢"
        color = "#2e7d32"
    
    # Display results
    st.markdown("---")
    st.markdown(f"## {emoji} Analysis Results")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:{color};">{status}</h3>
            <p style="color:#666;">Prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{probability:.1%}</h3>
            <p style="color:#666;">Churn Probability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:{color};">{risk}</h3>
            <p style="color:#666;">Risk Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        confidence = abs(probability - 0.5) * 2
        st.markdown(f"""
        <div class="metric-card">
            <h3>{confidence:.1%}</h3>
            <p style="color:#666;">Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability gauge
    st.markdown("---")
    st.markdown("## 📊 Churn Probability")
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(['Risk'], [probability], color=color, height=0.3)
    ax.set_xlim(0, 1)
    ax.axvline(x=0.4, color='orange', linestyle='--', alpha=0.7, label='Medium Risk (40%)')
    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High Risk (70%)')
    ax.text(probability, 0, f'{probability:.1%}', ha='center', va='center', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_title('Customer Churn Risk Assessment', fontweight='bold')
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
        good_factors.append("Engaged via newsletter")
    
    st.markdown(f"""
    <div class="{risk_class}">
        <strong>Risk Level: {risk}</strong><br>
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
        **Immediate Action Required:**
        - Send personalized discount offer (20-30% off)
        - Proactive customer outreach
        - Request feedback survey
        - Loyalty program invitation
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
        - Thank you note with offer
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
    # Welcome message
    st.info("👈 **Get Started**: Enter customer details in the sidebar and click 'Predict Churn Risk'")
    
    # Show features
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📊 Key Features
        - Churn probability scoring
        - Risk factor analysis
        - Actionable recommendations
        """)
    with col2:
        st.markdown("""
        ### 🎯 Risk Levels
        - 🟢 Low: <40% probability
        - 🟡 Medium: 40-70%
        - 🔴 High: >70%
        """)
    with col3:
        st.markdown("""
        ### 💡 Benefits
        - Reduce customer churn
        - Increase retention
        - Maximize lifetime value
        """)

# =========================
# FOOTER
# =========================
st.divider()
st.markdown("""
<div style="text-align: center; color: #999; padding: 20px;">
    <p>🤖 Powered by Machine Learning | 📊 Analyzes 10 customer behavior factors</p>
    <p style="font-size: 0.8rem;">Predictions are estimates based on behavioral patterns</p>
</div>
""", unsafe_allow_html=True)