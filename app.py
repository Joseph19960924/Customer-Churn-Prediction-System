"""
Customer Churn Prediction System
A clean, production-ready Streamlit application for predicting customer churn
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict, List

# Try importing ML packages gracefully
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🎯",
    layout="wide"
)

# Feature columns used by the model
FEATURE_COLUMNS = [
    "age", "total_orders", "total_spend_usd", "avg_order_value_usd",
    "days_since_last_purchase", "reviews_given", "avg_review_score",
    "returns_made", "wishlist_items", "newsletter_subscribed"
]

# Risk thresholds
RISK_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4
}

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

@st.cache_resource
def load_or_create_model():
    """Load existing model or create a demo model"""
    
    if not SKLEARN_AVAILABLE:
        return None, FEATURE_COLUMNS
    
    np.random.seed(42)
    
    # Generate synthetic training data
    n_samples = 1000
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
    
    # Create target variable based on business rules
    churn_score = (
        (X_train["days_since_last_purchase"] > 60) * 0.3 +
        (X_train["total_orders"] < 5) * 0.2 +
        (X_train["returns_made"] > X_train["total_orders"] * 0.2) * 0.2 +
        (X_train["avg_review_score"] < 2.5) * 0.15
    )
    y_train = (churn_score > 0.4).astype(int)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, FEATURE_COLUMNS


def predict_churn(model, features: pd.DataFrame) -> Tuple[float, int]:
    """Make churn prediction"""
    
    if model is not None and SKLEARN_AVAILABLE:
        probability = model.predict_proba(features)[0][1]
        prediction = model.predict(features)[0]
    else:
        # Rule-based fallback
        probability = _rule_based_prediction(features)
        prediction = 1 if probability > RISK_THRESHOLDS["medium"] else 0
    
    return probability, prediction


def _rule_based_prediction(features: pd.DataFrame) -> float:
    """Simple rule-based prediction when ML model is unavailable"""
    score = 0.0
    
    if features["days_since_last_purchase"].iloc[0] > 60:
        score += 0.3
    if features["total_orders"].iloc[0] < 5:
        score += 0.2
    if features["returns_made"].iloc[0] > features["total_orders"].iloc[0] * 0.2:
        score += 0.2
    if features["avg_review_score"].iloc[0] < 2.5:
        score += 0.15
    if features["newsletter_subscribed"].iloc[0] == 0:
        score += 0.05
    
    return min(score, 0.95)


# Initialize model
model, feature_columns = load_or_create_model()

# ============================================================================
# UI COMPONENTS
# ============================================================================

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .risk-high {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-left: 5px solid #c62828;
            padding: 1.5rem;
            border-radius: 10px;
        }
        .risk-medium {
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            border-left: 5px solid #f9a825;
            padding: 1.5rem;
            border-radius: 10px;
        }
        .risk-low {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-left: 5px solid #2e7d32;
            padding: 1.5rem;
            border-radius: 10px;
        }
        .footer {
            text-align: center;
            color: #999;
            padding: 2rem;
            margin-top: 2rem;
            border-top: 1px solid #ddd;
        }
        </style>
    """, unsafe_allow_html=True)


def create_sidebar_inputs() -> Dict:
    """Create input widgets in sidebar and return values"""
    
    with st.sidebar:
        st.markdown("## 📊 Customer Profile")
        st.markdown("---")
        
        # Demographics
        st.markdown("### 👤 Demographics")
        age = st.number_input("Age", 18, 100, 35)
        
        # Purchase Behavior
        st.markdown("### 🛍️ Purchase Behavior")
        col1, col2 = st.columns(2)
        with col1:
            total_orders = st.number_input("Total Orders", 0, 500, 15)
            avg_order_value = st.number_input("Avg Order Value ($)", 0, 1000, 50)
        with col2:
            total_spend = st.number_input("Total Spend ($)", 0, 100000, 750)
            days_since_last = st.number_input("Days Since Last Purchase", 0, 365, 30)
        
        # Engagement Metrics
        st.markdown("### ⭐ Engagement")
        col1, col2 = st.columns(2)
        with col1:
            reviews = st.number_input("Reviews Given", 0, 200, 5)
            returns = st.number_input("Returns Made", 0, 100, 1)
        with col2:
            rating = st.slider("Avg Review Rating", 0.0, 5.0, 3.5, 0.1)
            wishlist = st.number_input("Wishlist Items", 0, 200, 8)
        
        # Marketing Preferences
        st.markdown("### 📧 Marketing")
        newsletter = st.radio("Newsletter Subscribed", ["Yes", "No"], horizontal=True)
        newsletter_val = 1 if newsletter == "Yes" else 0
        
        st.markdown("---")
        predict_button = st.button("🎯 Predict Churn Risk", type="primary", use_container_width=True)
        
    return {
        "age": age,
        "total_orders": total_orders,
        "total_spend": total_spend,
        "avg_order_value": avg_order_value,
        "days_since_last": days_since_last,
        "reviews": reviews,
        "rating": rating,
        "returns": returns,
        "wishlist": wishlist,
        "newsletter_val": newsletter_val,
        "predict_button": predict_button
    }


def get_risk_level(probability: float) -> Tuple[str, str, str]:
    """Determine risk level based on probability"""
    
    if probability >= RISK_THRESHOLDS["high"]:
        return "High Risk", "risk-high", "🔴"
    elif probability >= RISK_THRESHOLDS["medium"]:
        return "Medium Risk", "risk-medium", "🟡"
    else:
        return "Low Risk", "risk-low", "🟢"


def analyze_risk_factors(inputs: Dict) -> Tuple[List[str], List[str]]:
    """Analyze risk and positive factors"""
    
    risk_factors = []
    positive_factors = []
    
    # Risk factors
    if inputs["days_since_last"] > 60:
        risk_factors.append(f"Long inactivity ({inputs['days_since_last']} days)")
    if inputs["total_orders"] < 5:
        risk_factors.append(f"Low purchase frequency ({inputs['total_orders']} orders)")
    if inputs["returns"] > inputs["total_orders"] * 0.2:
        risk_factors.append(f"High return rate ({inputs['returns']} returns)")
    if inputs["rating"] < 2.5:
        risk_factors.append("Low satisfaction ratings")
    if inputs["newsletter_val"] == 0:
        risk_factors.append("Not subscribed to newsletter")
    
    # Positive factors
    if inputs["days_since_last"] < 30:
        positive_factors.append("Recent purchase activity")
    if inputs["total_orders"] > 20:
        positive_factors.append(f"Loyal customer ({inputs['total_orders']} orders)")
    if inputs["rating"] > 4.0:
        positive_factors.append("High satisfaction ratings")
    if inputs["newsletter_val"] == 1:
        positive_factors.append("Engaged via newsletter")
    if inputs["wishlist"] > 10:
        positive_factors.append("Active interest (wishlist items)")
    
    return risk_factors, positive_factors


def display_metrics(probability: float, risk_level: str, risk_color: str):
    """Display key metrics in cards"""
    
    confidence = abs(probability - 0.5) * 2
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        (col1, risk_level, "Risk Level", risk_color),
        (col2, f"{probability:.1%}", "Churn Probability", "#333"),
        (col3, f"{confidence:.1%}", "Confidence", "#333"),
        (col4, "Will Churn" if probability >= 0.4 else "Will Stay", "Prediction", risk_color)
    ]
    
    for col, value, label, color in metrics:
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {color}; margin: 0;">{value}</h3>
                    <p style="color: #666; margin: 0.5rem 0 0 0;">{label}</p>
                </div>
            """, unsafe_allow_html=True)


def display_probability_gauge(probability: float, risk_color: str):
    """Display probability gauge chart"""
    
    fig, ax = plt.subplots(figsize=(10, 2))
    
    ax.barh(["Churn Risk"], [probability], color=risk_color, height=0.3, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", fontsize=10)
    ax.set_title("Churn Probability Assessment", fontweight="bold", pad=15)
    
    # Add threshold lines
    ax.axvline(x=0.4, color="orange", linestyle="--", alpha=0.7, linewidth=2, label="Medium Risk (40%)")
    ax.axvline(x=0.7, color="red", linestyle="--", alpha=0.7, linewidth=2, label="High Risk (70%)")
    
    # Add value label
    ax.text(probability, 0, f"{probability:.1%}", 
            ha="center", va="center", fontweight="bold", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    
    st.pyplot(fig)


def display_recommendations(probability: float, risk_factors: List[str]):
    """Display actionable recommendations"""
    
    st.markdown("---")
    st.markdown("## 💡 Recommendations")
    
    if probability >= 0.7:
        st.error("""
        **🚨 Immediate Actions Required:**
        - Send personalized discount offer (20-30% off)
        - Proactive customer outreach via phone/email
        - Request feedback survey with incentive
        - Invite to loyalty program with bonus points
        - Schedule re-engagement campaign
        """)
    elif probability >= 0.4:
        st.warning("""
        **⚠️ Preventive Actions:**
        - Send personalized product recommendations
        - Invite to loyalty program
        - Share educational content
        - Offer free shipping on next purchase
        - Request product reviews
        """)
    else:
        st.success("""
        **✅ Growth Actions:**
        - Upsell and cross-sell opportunities
        - Referral program invitation
        - VIP program consideration
        - Send thank you note with special offer
        - Early access to new products
        """)


def display_value_analysis(total_spend: float, probability: float):
    """Display customer value analysis"""
    
    st.markdown("---")
    st.markdown("## 💰 Value Analysis")
    
    estimated_clv = total_spend * (1 - probability) * 2
    retention_potential = (1 - probability) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Value", f"${total_spend:,.0f}")
    with col2:
        st.metric("Est. Lifetime Value", f"${estimated_clv:,.0f}")
    with col3:
        st.metric("Retention Potential", f"{retention_potential:.0f}%")


def display_welcome_message():
    """Display welcome message when no prediction has been made"""
    
    st.info("👈 **Get Started**: Enter customer details in the sidebar and click 'Predict Churn Risk'")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Analysis
        - Churn probability scoring
        - Risk factor identification
        - Confidence metrics
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
        ### 💡 Value
        - Reduce customer churn
        - Increase retention
        - Maximize CLV
        """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Apply styling
    apply_custom_css()
    
    # Header
    st.markdown('<p class="main-header">🎯 Customer Churn Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Customer Retention Analytics</p>', unsafe_allow_html=True)
    st.divider()
    
    # Get user inputs
    inputs = create_sidebar_inputs()
    
    # Main logic
    if inputs["predict_button"]:
        
        # Prepare features
        features = pd.DataFrame([{
            "age": inputs["age"],
            "total_orders": inputs["total_orders"],
            "total_spend_usd": inputs["total_spend"],
            "avg_order_value_usd": inputs["avg_order_value"],
            "days_since_last_purchase": inputs["days_since_last"],
            "reviews_given": inputs["reviews"],
            "avg_review_score": inputs["rating"],
            "returns_made": inputs["returns"],
            "wishlist_items": inputs["wishlist"],
            "newsletter_subscribed": inputs["newsletter_val"]
        }])
        
        features = features[feature_columns]
        
        # Make prediction
        with st.spinner("Analyzing customer behavior..."):
            probability, _ = predict_churn(model, features)
        
        # Get risk level
        risk_level, risk_class, risk_emoji = get_risk_level(probability)
        risk_color = {"High Risk": "#c62828", "Medium Risk": "#f9a825", "Low Risk": "#2e7d32"}[risk_level]
        
        # Display results
        st.markdown(f"## {risk_emoji} Analysis Results")
        
        # Display metrics
        display_metrics(probability, risk_level, risk_color)
        
        # Display probability gauge
        st.markdown("---")
        st.markdown("## 📊 Risk Analysis")
        display_probability_gauge(probability, risk_color)
        
        # Analyze risk factors
        risk_factors, positive_factors = analyze_risk_factors(inputs)
        
        # Display risk assessment
        st.markdown("---")
        st.markdown("## 🎯 Risk Assessment")
        
        st.markdown(f"""
            <div class="{risk_class}">
                <strong>Risk Level: {risk_level}</strong><br>
                Churn Probability: {probability:.1%}<br><br>
                
                <strong>⚠️ Risk Factors ({len(risk_factors)}):</strong><br>
                {chr(10).join(['• ' + f for f in risk_factors]) if risk_factors else '✅ No significant risk factors identified'}<br><br>
                
                <strong>✅ Positive Factors ({len(positive_factors)}):</strong><br>
                {chr(10).join(['• ' + f for f in positive_factors]) if positive_factors else 'ℹ️ No significant positive factors identified'}
            </div>
        """, unsafe_allow_html=True)
        
        # Display recommendations
        display_recommendations(probability, risk_factors)
        
        # Display value analysis
        display_value_analysis(inputs["total_spend"], probability)
        
    else:
        display_welcome_message()
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>🤖 Powered by Machine Learning | 📊 Analyzes 10 customer behavior factors</p>
            <p style="font-size: 0.8rem;">Predictions are estimates based on behavioral patterns | Use with business judgment</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()