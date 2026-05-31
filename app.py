import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

try:
    model, model_columns = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f7fa;
        }
        
        .main-header {
            font-size: 2rem;
            font-weight: bold;
            color: #0b1f3a;
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
# HEADER
# =========================
st.markdown('<p class="main-header">Customer Churn Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Customer Retention Analytics</p>', unsafe_allow_html=True)

st.divider()

# =========================
# SIDEBAR INPUTS
# =========================
with st.sidebar:
    st.markdown("### Customer Profile")
    st.markdown("---")
    
    st.markdown("**Demographics**")
    age = st.number_input("Age", 18, 100, 30, help="Customer age in years")
    
    st.markdown("---")
    st.markdown("**Purchase Behavior**")
    total_orders = st.number_input("Total Orders", 0, 1000, 10, help="Lifetime number of orders")
    total_spend = st.number_input("Total Spend (USD)", 0.0, 100000.0, 500.0, help="Lifetime spend in USD")
    avg_order_value = st.number_input("Average Order Value (USD)", 0.0, 10000.0, 50.0, help="Average value per order")
    days_since_last_purchase = st.number_input("Days Since Last Purchase", 0, 1000, 30, help="Days since customer last ordered")
    
    st.markdown("---")
    st.markdown("**Engagement Metrics**")
    reviews_given = st.number_input("Reviews Given", 0, 500, 5, help="Number of product reviews written")
    avg_review_score = st.slider("Average Review Score", 0.0, 5.0, 3.5, 0.1, help="Average rating given (1-5 stars)")
    returns_made = st.number_input("Returns Made", 0, 100, 0, help="Number of products returned")
    wishlist_items = st.number_input("Wishlist Items", 0, 500, 10, help="Number of items saved to wishlist")
    
    st.markdown("---")
    st.markdown("**Marketing Preferences**")
    newsletter_subscribed = st.radio("Newsletter Subscribed", ["Yes", "No"], horizontal=True)
    newsletter_value = 1 if newsletter_subscribed == "Yes" else 0
    
    st.markdown("---")
    st.caption("Powered by Machine Learning")
    st.caption("Model analyzes 10 customer behavior factors")

# =========================
# PREDICTION BUTTON
# =========================
if st.button("Analyze Customer Risk", type="primary", use_container_width=True):
    
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
    input_data = input_data.reindex(columns=model_columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Determine risk level
    if probability >= 0.7:
        risk_level = "High Risk"
        risk_color = "#c62828"
        risk_class = "risk-high"
        churn_status = "Will Churn"
    elif probability >= 0.4:
        risk_level = "Medium Risk"
        risk_color = "#f9a825"
        risk_class = "risk-medium"
        churn_status = "At Risk"
    else:
        risk_level = "Low Risk"
        risk_color = "#2e7d32"
        risk_class = "risk-low"
        churn_status = "Will Stay"
    
    # Display Results
    st.markdown("---")
    st.markdown("## Analysis Results")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:{risk_color}; margin:0;">{churn_status}</h3>
            <p style="margin:0; color:#666;">Prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0;">{probability:.1%}</h3>
            <p style="margin:0; color:#666;">Churn Probability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0;">{risk_level}</h3>
            <p style="margin:0; color:#666;">Risk Category</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        confidence = abs(probability - 0.5) * 2
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0;">{confidence:.1%}</h3>
            <p style="margin:0; color:#666;">Confidence Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability Gauge Chart
    st.markdown("---")
    st.markdown("## Churn Probability Analysis")
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Create horizontal gauge
    colors_gauge = ['#2e7d32', '#f9a825', '#c62828']
    ax.barh(['Churn Probability'], [probability], color=risk_color, height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title('Customer Churn Risk Assessment', fontsize=12, fontweight='bold')
    
    # Add threshold lines
    ax.axvline(x=0.4, color='orange', linestyle='--', alpha=0.7, label='Medium Risk Threshold (40%)')
    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold (70%)')
    
    # Add value text
    ax.text(probability, 0, f'{probability:.1%}', 
           ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.legend(loc='lower right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Risk Assessment Details
    st.markdown("---")
    st.markdown("## Risk Assessment")
    
    # Generate risk factors based on inputs
    risk_factors = []
    positive_factors = []
    
    if days_since_last_purchase > 60:
        risk_factors.append(f"Long time since last purchase ({days_since_last_purchase} days)")
    elif days_since_last_purchase < 30:
        positive_factors.append(f"Recent purchase activity ({days_since_last_purchase} days ago)")
    
    if returns_made > total_orders * 0.2:
        risk_factors.append(f"High return rate ({returns_made} returns out of {total_orders} orders)")
    
    if reviews_given < 2 and total_orders > 5:
        risk_factors.append("Low engagement -很少 product reviews")
    else:
        positive_factors.append(f"Active reviewer ({reviews_given} reviews given)")
    
    if avg_order_value < 20 and total_spend > 500:
        risk_factors.append("Low average order value relative to total spend")
    
    if wishlist_items > 20:
        positive_factors.append(f"High interest shown ({wishlist_items} items in wishlist)")
    
    if newsletter_value == 1:
        positive_factors.append("Subscribed to newsletter - good engagement channel")
    else:
        risk_factors.append("Not subscribed to newsletter - missed engagement opportunity")
    
    if total_orders < 3:
        risk_factors.append("Low purchase frequency - limited customer history")
    elif total_orders > 20:
        positive_factors.append(f"Loyal customer with {total_orders} orders")
    
    st.markdown(f"""
    <div class="{risk_class}">
        <strong>Risk Level: {risk_level}</strong><br>
        Churn Probability: {probability:.1%}<br><br>
        
        <strong>Risk Factors:</strong><br>
        {chr(10).join(['• ' + f for f in risk_factors]) if risk_factors else '• No significant risk factors identified'}<br><br>
        
        <strong>Positive Factors:</strong><br>
        {chr(10).join(['• ' + p for p in positive_factors]) if positive_factors else '• No significant positive factors identified'}
    </div>
    """, unsafe_allow_html=True)
    
    # Actionable Recommendations
    st.markdown("---")
    st.markdown("## Recommended Actions")
    
    if probability >= 0.7:
        st.markdown("""
        <div class="insight-box">
            <strong>Immediate Retention Actions Required:</strong><br><br>
            • Send personalized discount offer (20-30% off)<br>
            • Assign dedicated account manager for high-value customers<br>
            • Request feedback through survey or call<br>
            • Offer loyalty program enrollment with bonus points<br>
            • Send re-engagement email campaign with compelling offer
        </div>
        """, unsafe_allow_html=True)
        
        # Specific recommendations based on factors
        if days_since_last_purchase > 60:
            st.info("Action: Send win-back campaign with time-sensitive discount")
        if returns_made > total_orders * 0.2:
            st.info("Action: Review return reasons and improve product quality/descriptions")
        if newsletter_value == 0:
            st.info("Action: Offer incentive to subscribe to newsletter (10% off next purchase)")
    
    elif probability >= 0.4:
        st.markdown("""
        <div class="insight-box">
            <strong>Preventive Retention Actions:</strong><br><br>
            • Send automated personalized product recommendations<br>
            • Invite to loyalty program with tiered benefits<br>
            • Request product review with small incentive<br>
            • Send educational content about product usage<br>
            • Offer free shipping on next purchase
        </div>
        """, unsafe_allow_html=True)
        
        if days_since_last_purchase > 30:
            st.info("Action: Send reminder email with popular products in their category")
        if avg_review_score < 3.0:
            st.info("Action: Follow up on low ratings to resolve issues")
    
    else:
        st.markdown("""
        <div class="insight-box">
            <strong>Retention & Growth Actions:</strong><br><br>
            • Upsell and cross-sell relevant products<br>
            • Invite to VIP program or early access<br>
            • Request referral from satisfied customer<br>
            • Send personalized thank you note<br>
            • Offer subscription or membership program
        </div>
        """, unsafe_allow_html=True)
        
        if positive_factors:
            st.success(f"Leverage strengths: {', '.join(positive_factors[:2])}")
    
    # Customer Lifetime Value Estimate
    st.markdown("---")
    st.markdown("## Customer Value Analysis")
    
    # Calculate estimated CLV
    avg_order_freq = total_orders / max(1, (days_since_last_purchase / 30)) if days_since_last_purchase > 0 else total_orders
    estimated_clv = total_spend * (1 - probability) * 2  # Simple CLV estimate
    
    col_clv1, col_clv2, col_clv3 = st.columns(3)
    
    with col_clv1:
        st.metric("Total Customer Value", f"${total_spend:,.2f}")
    
    with col_clv2:
        st.metric("Estimated Lifetime Value", f"${estimated_clv:,.2f}",
                 delta=f"{'Positive' if estimated_clv > total_spend else 'At Risk'}")
    
    with col_clv3:
        retention_potential = (1 - probability) * 100
        st.metric("Retention Potential", f"{retention_potential:.0f}%")
    
    # Customer Segments
    st.markdown("---")
    st.markdown("## Customer Segment")
    
    # Determine customer segment based on metrics
    if total_orders > 20 and total_spend > 2000:
        segment = "VIP / High Value"
        segment_color = "#2e7d32"
    elif total_orders > 10 and total_spend > 1000:
        segment = "Regular / Medium Value"
        segment_color = "#1976d2"
    elif total_orders > 3:
        segment = "Occasional / Low Value"
        segment_color = "#f9a825"
    else:
        segment = "New / Trial"
        segment_color = "#9e9e9e"
    
    st.markdown(f"""
    <div style="background-color: white; padding: 20px; border-radius: 10px; text-align: center;">
        <h3 style="color: {segment_color};">{segment} Customer</h3>
        <p style="color: #666;">Based on {total_orders} orders totaling ${total_spend:,.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Engagement Metrics Visualization
    st.markdown("---")
    st.markdown("## Engagement Metrics")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Purchase frequency vs review activity
    categories = ['Orders', 'Reviews', 'Returns', 'Wishlist']
    values = [total_orders, reviews_given, returns_made, wishlist_items]
    colors_bar = ['#4CAF50', '#2196F3', '#F44336', '#FF9800']
    
    bars = ax1.bar(categories, values, color=colors_bar, alpha=0.7)
    ax1.set_ylabel('Count')
    ax1.set_title('Customer Activity Metrics', fontsize=12, fontweight='bold')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # Engagement score gauge
    engagement_score = min(100, (
        (min(total_orders, 50) / 50 * 30) +
        (min(reviews_given, 20) / 20 * 20) +
        (max(0, 100 - min(days_since_last_purchase, 365) / 365 * 30)) +
        (newsletter_value * 10) +
        (min(wishlist_items, 20) / 20 * 10)
    ))
    
    ax2.barh(['Engagement Score'], [engagement_score], color='#1976d2', height=0.5)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Score')
    ax2.set_title(f'Customer Engagement Score: {engagement_score:.0f}/100', fontsize=12, fontweight='bold')
    ax2.axvline(x=70, color='green', linestyle='--', alpha=0.7, label='High Engagement')
    ax2.axvline(x=40, color='orange', linestyle='--', alpha=0.7, label='Medium Engagement')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

# =========================
# FOOTER / INFO SECTION
# =========================
with st.expander("About This System", expanded=False):
    st.markdown("""
    **How Churn Prediction Works:**
    
    This system analyzes 10 customer behavior factors to predict churn risk:
    
    | Factor | Impact on Churn |
    |--------|-----------------|
    | Days since last purchase | High - Long gaps indicate disengagement |
    | Order frequency | High - Frequent buyers are loyal |
    | Return rate | Medium - High returns suggest dissatisfaction |
    | Review activity | Medium - Engaged customers leave reviews |
    | Newsletter subscription | Low - Subscribers are easier to reach |
    | Wishlist items | Low - Indicates continued interest |
    
    **Churn Risk Levels:**
    - Low Risk (<40%): Stable customer, likely to stay
    - Medium Risk (40-70%): At risk, needs engagement
    - High Risk (>70%): Likely to churn, immediate action needed
    
    **Recommended Actions by Risk Level:**
    - Low Risk: Upsell, cross-sell, loyalty programs
    - Medium Risk: Engagement campaigns, feedback requests
    - High Risk: Discount offers, win-back campaigns
    """)

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