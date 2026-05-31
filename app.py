"""
Customer Churn Prediction System
Complete Application with all features
"""

import streamlit as st
import pandas as pd
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

# =========================
# DEPENDENCY CHECK AND INSTALLATION
# =========================
def check_and_install_dependencies():
    """Check if required packages are installed and provide helpful error messages"""
    required_packages = {
        'joblib': 'joblib',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    for package, install_name in required_packages.items():
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(install_name)
    
    if missing_packages:
        st.error(f"""
        ### Missing Dependencies Detected
        
        The following packages are required but not installed:
        {', '.join(missing_packages)}
        
        Please install them using:
        pip install {' '.join(missing_packages)}
        
        Or create a requirements.txt file.
        """)
        st.stop()

# Run dependency check first
check_and_install_dependencies()

# Now import all packages
try:
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    from sklearn.ensemble import RandomForestClassifier
    import base64
    from io import BytesIO
except ImportError as e:
    st.error(f"Failed to import required packages: {str(e)}")
    st.stop()

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
# MODEL LOADING WITH ERROR HANDLING
# =========================
@st.cache_resource
def load_model():
    """Load the trained model and feature columns with error handling"""
    try:
        import os
        if not os.path.exists("churn_model.pkl"):
            st.warning("Model file not found. Creating a sample model...")
            return create_sample_model()
        
        if not os.path.exists("model_columns.pkl"):
            st.warning("Model columns file not found. Creating sample columns...")
            return create_sample_model()
        
        model = joblib.load("churn_model.pkl")
        model_columns = joblib.load("model_columns.pkl")
        return model, model_columns
    except Exception as e:
        st.warning(f"Error loading model: {str(e)}. Creating sample model...")
        return create_sample_model()

def create_sample_model():
    """Create a sample model for demonstration purposes"""
    from sklearn.ensemble import RandomForestClassifier
    
    # Create sample training data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    sample_data = pd.DataFrame({
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
    
    # Create synthetic target (churn)
    churn_prob = (
        (sample_data['days_since_last_purchase'] > 60) * 0.3 +
        (sample_data['total_orders'] < 5) * 0.2 +
        (sample_data['returns_made'] > sample_data['total_orders'] * 0.2) * 0.2 +
        (sample_data['avg_review_score'] < 2.5) * 0.15 +
        (sample_data['newsletter_subscribed'] == 0) * 0.05
    )
    sample_data['churn'] = (churn_prob > np.random.uniform(0, 0.8, n_samples)).astype(int)
    
    # Train a simple model
    feature_columns = [col for col in sample_data.columns if col != 'churn']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(sample_data[feature_columns], sample_data['churn'])
    
    st.success("Sample model created successfully for demonstration!")
    st.info("Note: Using a demo model. For production, please train and save your own model.")
    
    return model, feature_columns

try:
    model, model_columns = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to initialize the model: {str(e)}")
    st.stop()

# =========================
# CUSTOM CSS STYLES
# =========================
st.markdown("""
    <style>
        .main-header {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        
        .sub-header {
            font-size: 1rem;
            color: #666;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            text-align: center;
        }
        
        .risk-high {
            background-color: #ffebee;
            border-left: 4px solid #c62828;
            padding: 15px;
            border-radius: 8px;
        }
        
        .risk-medium {
            background-color: #fff3e0;
            border-left: 4px solid #f9a825;
            padding: 15px;
            border-radius: 8px;
        }
        
        .risk-low {
            background-color: #e8f5e9;
            border-left: 4px solid #2e7d32;
            padding: 15px;
            border-radius: 8px;
        }
        
        .insight-box {
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #1976d2;
            margin: 20px 0;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER SECTION
# =========================
st.markdown('<p class="main-header">🎯 Customer Churn Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Customer Retention Analytics</p>', unsafe_allow_html=True)

st.divider()

# =========================
# SIDEBAR - INPUT SECTION
# =========================
with st.sidebar:
    st.markdown("### 📊 Customer Profile")
    st.markdown("---")
    
    st.markdown("**👤 Demographics**")
    age = st.number_input("Age", 18, 100, 30)
    
    st.markdown("**🛍️ Purchase Behavior**")
    total_orders = st.number_input("Total Orders", 0, 1000, 10)
    total_spend = st.number_input("Total Spend (USD)", 0.0, 100000.0, 500.0)
    avg_order_value = st.number_input("Average Order Value (USD)", 0.0, 10000.0, 50.0)
    days_since_last_purchase = st.number_input("Days Since Last Purchase", 0, 1000, 30)
    
    st.markdown("**⭐ Engagement Metrics**")
    reviews_given = st.number_input("Reviews Given", 0, 500, 5)
    avg_review_score = st.slider("Average Review Score", 0.0, 5.0, 3.5, 0.1)
    returns_made = st.number_input("Returns Made", 0, 100, 0)
    wishlist_items = st.number_input("Wishlist Items", 0, 500, 10)
    
    st.markdown("**📧 Marketing Preferences**")
    newsletter_subscribed = st.radio("Newsletter Subscribed", ["Yes", "No"], horizontal=True)
    newsletter_value = 1 if newsletter_subscribed == "Yes" else 0
    
    st.markdown("---")
    analyze_button = st.button("🎯 Analyze Customer Risk", type="primary", use_container_width=True)

# =========================
# MAIN CONTENT AREA
# =========================
if analyze_button:
    with st.spinner("Analyzing customer behavior..."):
        # Prepare input data
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
        
        # Align with model columns
        try:
            input_data = input_data.reindex(columns=model_columns, fill_value=0)
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.stop()
    
    # Determine risk level
    if probability >= 0.7:
        risk_level = "High Risk"
        risk_color = "#c62828"
        risk_class = "risk-high"
        churn_status = "⚠️ Will Churn"
    elif probability >= 0.4:
        risk_level = "Medium Risk"
        risk_color = "#f9a825"
        risk_class = "risk-medium"
        churn_status = "⚠️ At Risk"
    else:
        risk_level = "Low Risk"
        risk_color = "#2e7d32"
        risk_class = "risk-low"
        churn_status = "✅ Will Stay"
    
    # Display Results
    st.markdown("---")
    st.markdown("## Analysis Results")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:{risk_color};">{churn_status}</h3>
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
            <p>Risk Category</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        confidence = abs(probability - 0.5) * 2
        st.markdown(f"""
        <div class="metric-card">
            <h3>{confidence:.1%}</h3>
            <p>Confidence Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability Chart
    st.markdown("---")
    st.markdown("## Churn Probability Analysis")
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.barh(['Churn Probability'], [probability], color=risk_color, height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title('Customer Churn Risk Assessment', fontsize=12, fontweight='bold')
    ax.axvline(x=0.4, color='orange', linestyle='--', alpha=0.7, label='Medium Risk (40%)')
    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High Risk (70%)')
    ax.text(probability, 0, f'{probability:.1%}', ha='center', va='bottom', fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Risk Assessment Details
    st.markdown("---")
    st.markdown("## Risk Assessment")
    
    # Generate risk factors
    risk_factors = []
    positive_factors = []
    
    if days_since_last_purchase > 60:
        risk_factors.append(f"Long time since last purchase ({days_since_last_purchase} days)")
    elif days_since_last_purchase < 30:
        positive_factors.append(f"Recent purchase activity ({days_since_last_purchase} days ago)")
    
    if returns_made > total_orders * 0.2 and total_orders > 0:
        risk_factors.append(f"High return rate ({returns_made} returns)")
    
    if reviews_given < 2 and total_orders > 5:
        risk_factors.append("Low engagement - few product reviews")
    elif reviews_given > 10:
        positive_factors.append(f"Active reviewer ({reviews_given} reviews)")
    
    if newsletter_value == 1:
        positive_factors.append("Subscribed to newsletter")
    else:
        risk_factors.append("Not subscribed to newsletter")
    
    if total_orders < 3:
        risk_factors.append("Low purchase frequency")
    elif total_orders > 20:
        positive_factors.append(f"Loyal customer with {total_orders} orders")
    
    st.markdown(f"""
    <div class="{risk_class}">
        <strong>Risk Level: {risk_level}</strong><br>
        Churn Probability: {probability:.1%}<br><br>
        
        <strong>Risk Factors:</strong><br>
        {chr(10).join(['• ' + f for f in risk_factors]) if risk_factors else '• No significant risk factors'}<br><br>
        
        <strong>Positive Factors:</strong><br>
        {chr(10).join(['• ' + p for p in positive_factors]) if positive_factors else '• No significant positive factors'}
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("---")
    st.markdown("## Recommended Actions")
    
    if probability >= 0.7:
        st.markdown("""
        <div class="insight-box">
            <strong>Immediate Retention Actions Required:</strong><br><br>
            • Send personalized discount offer (20-30% off)<br>
            • Assign dedicated account manager<br>
            • Request feedback through survey or call<br>
            • Offer loyalty program enrollment<br>
            • Send re-engagement email campaign
        </div>
        """, unsafe_allow_html=True)
    elif probability >= 0.4:
        st.markdown("""
        <div class="insight-box">
            <strong>Preventive Retention Actions:</strong><br><br>
            • Send personalized product recommendations<br>
            • Invite to loyalty program<br>
            • Request product review with incentive<br>
            • Send educational content<br>
            • Offer free shipping on next purchase
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="insight-box">
            <strong>Retention & Growth Actions:</strong><br><br>
            • Upsell and cross-sell relevant products<br>
            • Invite to VIP program<br>
            • Request referral from satisfied customer<br>
            • Send personalized thank you note<br>
            • Offer subscription program
        </div>
        """, unsafe_allow_html=True)
    
    # Customer Value Analysis
    st.markdown("---")
    st.markdown("## Customer Value Analysis")
    
    estimated_clv = total_spend * (1 - probability) * 2
    retention_potential = (1 - probability) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customer Value", f"${total_spend:,.2f}")
    with col2:
        st.metric("Estimated Lifetime Value", f"${estimated_clv:,.2f}")
    with col3:
        st.metric("Retention Potential", f"{retention_potential:.0f}%")

else:
    st.info("👈 Please enter customer details in the sidebar and click 'Analyze Customer Risk'")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Powered by Machine Learning | Analyzes 10 customer behavior patterns</p>
    <p style="font-size: 0.8rem;">Predictions are estimates based on historical patterns. Use with business judgment.</p>
</div>
""", unsafe_allow_html=True)