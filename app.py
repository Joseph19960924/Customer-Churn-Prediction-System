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
        ### ❌ Missing Dependencies Detected
        
        The following packages are required but not installed:
        **{', '.join(missing_packages)}**
        
        **Please install them using:**
        ```bash
        pip install {' '.join(missing_packages)}