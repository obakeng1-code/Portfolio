"""
Credit Card Fraud Detection - Streamlit Web Application
Interactive interface for real-time fraud detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from main import FraudDetectionModel
import logging
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fraud-high { color: #ff4444; font-weight: bold; }
    .fraud-mid { color: #ffaa00; font-weight: bold; }
    .fraud-low { color: #00aa00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def load_model():
    """Load the fraud detection model (cached for efficiency)."""
    try:
        return FraudDetectionModel()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Header
st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    prediction_mode = st.radio(
        "Select Prediction Mode",
        ["Single Transaction", "Batch Upload", "Demo"]
    )
    
    threshold = st.slider(
        "Fraud Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Score above this will be classified as Fraud"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Decision Thresholds")
    st.info("""
    - **Auto-Approve**: Score < 30%
    - **Verify (MFA)**: Score 30-85%
    - **Block**: Score > 85%
    """)

# Load model
detector = load_model()

if detector is None:
    st.error("❌ Failed to load the fraud detection model. Please ensure model files exist.")
    st.stop()

# Main content
if prediction_mode == "Single Transaction":
    st.header("🔍 Single Transaction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Enter Transaction Features")
        
        # Create input fields for all features
        input_data = {}
        feature_cols = st.columns(4)
        
        for idx, feature in enumerate(detector.feature_order):
            col_idx = idx % 4
            with feature_cols[col_idx]:
                input_data[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    step=0.01,
                    format="%.2f"
                )
    
    with col2:
        st.subheader("Prediction Result")
        
        if st.button("🔎 Analyze Transaction", use_container_width=True):
            try:
                # Create DataFrame from input
                transaction_df = pd.DataFrame([input_data])
                
                # Get prediction
                probs = detector.predict(transaction_df)
                risk_score = probs[0] * 100
                
                # Determine action
                action = detector.categorize_risk([risk_score])[0]
                
                # Display results
                col_metric1, col_metric2 = st.columns(2)
                
                with col_metric1:
                    st.metric(
                        "Risk Score",
                        f"{risk_score:.2f}%",
                        delta=None
                    )
                
                with col_metric2:
                    if action == "Block":
                        st.metric("Recommendation", "🛑 BLOCK")
                    elif action == "Verify":
                        st.metric("Recommendation", "⚠️ VERIFY")
                    else:
                        st.metric("Recommendation", "✅ APPROVE")
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Risk Score"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "#90EE90"},
                            {'range': [30, 85], 'color': "#FFD700"},
                            {'range': [85, 100], 'color': "#FF6B6B"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                st.markdown("---")
                st.subheader("📋 Interpretation")
                
                if risk_score < 30:
                    st.success(f"✅ **Low Risk** - This transaction has a {risk_score:.1f}% fraud probability. It can be auto-approved.")
                elif risk_score < 85:
                    st.warning(f"⚠️ **Medium Risk** - This transaction has a {risk_score:.1f}% fraud probability. Consider multi-factor authentication.")
                else:
                    st.error(f"🛑 **High Risk** - This transaction has a {risk_score:.1f}% fraud probability. It should be blocked immediately.")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

elif prediction_mode == "Batch Upload":
    st.header("📁 Batch Transaction Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload CSV with transactions",
        type=['csv'],
        help="CSV should contain all required features"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10))
            
            if st.button("🔍 Analyze Batch", use_container_width=True):
                try:
                    # Get predictions
                    probs = detector.predict(df)
                    risk_scores = probs * 100
                    actions = detector.categorize_risk(risk_scores)
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame({
                        'Risk_Score': risk_scores.round(2),
                        'Recommendation': actions,
                        'Probability': probs.round(4)
                    })
                    
                    # Display statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Transactions", len(df))
                    with col2:
                        frauds = sum(results_df['Recommendation'] == 'Block')
                        st.metric("High Risk (Block)", frauds)
                    with col3:
                        verify = sum(results_df['Recommendation'] == 'Verify')
                        st.metric("Medium Risk (Verify)", verify)
                    with col4:
                        safe = sum(results_df['Recommendation'] == 'Auto-Approve')
                        st.metric("Low Risk (Approve)", safe)
                    
                    st.markdown("---")
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df)
                    
                    # Risk distribution chart
                    st.subheader("📈 Risk Distribution")
                    
                    fig_dist = px.histogram(
                        results_df,
                        x='Risk_Score',
                        nbins=50,
                        title='Transaction Risk Score Distribution',
                        labels={'Risk_Score': 'Risk Score (%)', 'count': 'Number of Transactions'}
                    )
                    fig_dist.add_vline(x=30, line_dash="dash", line_color="orange", 
                                      annotation_text="Verify Threshold")
                    fig_dist.add_vline(x=85, line_dash="dash", line_color="red", 
                                      annotation_text="Block Threshold")
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Recommendation breakdown
                    st.subheader("🎯 Recommendation Breakdown")
                    recommendation_counts = results_df['Recommendation'].value_counts()
                    fig_rec = px.pie(
                        values=recommendation_counts.values,
                        names=recommendation_counts.index,
                        title='Transaction Distribution by Recommendation'
                    )
                    st.plotly_chart(fig_rec, use_container_width=True)
                    
                    # Download results
                    st.markdown("---")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error during batch prediction: {e}")
        
        except Exception as e:
            st.error(f"Error loading file: {e}")

elif prediction_mode == "Demo":
    st.header("🎮 Demo Mode")
    
    st.markdown("""
    ### Try Pre-Built Examples
    Click on an example below to see how the model responds to different transaction patterns.
    """)
    
    # Create demo transactions
    demo_options = {
        "Normal Transaction": {col: 0.0 for col in detector.feature_order},
        "Suspicious Pattern": {col: 2.0 if col in ['V4', 'V12', 'V14'] else 0.0 
                               for col in detector.feature_order},
        "High Risk Pattern": {col: -3.0 if col in ['V4', 'V10', 'V14', 'V17'] else 1.0 
                             for col in detector.feature_order},
    }
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (demo_name, demo_data) in enumerate(demo_options.items()):
        with [col1, col2, col3][idx]:
            if st.button(f"▶️ {demo_name}", use_container_width=True):
                # Predict
                demo_df = pd.DataFrame([demo_data])
                prob = detector.predict(demo_df)[0]
                risk_score = prob * 100
                action = detector.categorize_risk([risk_score])[0]
                
                # Display
                st.subheader(demo_name)
                
                # Color-coded risk indicator
                if risk_score < 30:
                    st.success(f"Risk: {risk_score:.1f}%")
                elif risk_score < 85:
                    st.warning(f"Risk: {risk_score:.1f}%")
                else:
                    st.error(f"Risk: {risk_score:.1f}%")
                
                st.info(f"**Action**: {action}")

# Footer
st.markdown("---")
st.markdown("""
### 📌 About This System
- **Model**: Random Forest Classifier trained on credit card transactions
- **Features**: 30 features including PCA components and transaction characteristics
- **Accuracy**: Trained with SMOTE to handle class imbalance
- **Thresholds**: Customizable fraud detection thresholds

*Last updated: February 10, 2026*
""")