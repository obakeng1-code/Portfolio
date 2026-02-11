"""
Credit Card Fraud Detection - Main Inference Script
Loads trained models and provides fraud prediction capabilities
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os

# Configure paths relative to this file for deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, 'fraud_model.pkl')
DEFAULT_SCALER = os.path.join(BASE_DIR, 'scaler.pkl')
DEFAULT_FEATURES = os.path.join(BASE_DIR, 'feature_order.pkl')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDetectionModel:
    def __init__(self, model_path=DEFAULT_MODEL, scaler_path=DEFAULT_SCALER, features_path=DEFAULT_FEATURES):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.features_path = Path(features_path)
        
        self.model = None
        self.scaler = None
        self.feature_order = None
        
        self.load_artifacts()

    def load_artifacts(self):
        """Load model, scaler, and feature order from disk."""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"✅ Model loaded from {self.model_path}")
            
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✅ Scaler loaded from {self.scaler_path}")
            
            if self.features_path.exists():
                self.feature_order = joblib.load(self.features_path)
                logger.info(f"✅ Feature order loaded from {self.features_path}")
        except Exception as e:
            logger.error(f"❌ Error loading artifacts: {e}")
            raise

    def preprocess(self, data):
        """
        Fixes the mismatch between CSV column names and model expectations.
        """
        df = data.copy()

        # 1. Drop 'Class' if it exists (extra column from Kaggle datasets)
        if 'Class' in df.columns:
            df = df.drop(columns=['Class'])

        # 2. Rename 'Amount' and 'Time' to the scaled versions expected by the model
        rename_map = {'Amount': 'scaled_amount', 'Time': 'scaled_time'}
        df = df.rename(columns=rename_map)

        # 3. Handle Scaling: If the input values are raw, apply the scaler
        # This is critical to get high risk scores above 80%
        try:
            cols_to_scale = ['scaled_amount', 'scaled_time']
            if all(col in df.columns for col in cols_to_scale):
                df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        except Exception as e:
            logger.warning(f"Scaling skipped or failed: {e}")

        # 4. Feature Alignment: Ensure all expected features exist and are in order
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0.0  # Fill missing columns with baseline

        # Reorder to exactly match the training data feature order
        df = df[self.feature_order]
        
        return df

    def predict(self, data):
        """Preprocesses data and returns probability scores (0 to 1)."""
        processed_data = self.preprocess(data)
        # Returns the probability of the Fraud class (index 1)
        return self.model.predict_proba(processed_data)[:, 1]

    def predict_batch(self, data, threshold=0.5):
        """Predict fraud for a batch with labels."""
        try:
            probs = self.predict(data)
            predictions = (probs >= threshold).astype(int)
            
            return pd.DataFrame({
                'Risk_Score': (probs * 100).round(2),
                'Prediction': predictions,
                'Prediction_Label': ['Fraud' if p == 1 else 'Normal' for p in predictions]
            })
        except Exception as e:
            logger.error(f"❌ Batch prediction error: {e}")
            raise

    def categorize_risk(self, scores):
        """Categorize risk scores into business action categories."""
        categories = []
        for score in scores:
            if score < 30: categories.append('Auto-Approve')
            elif score < 85: categories.append('Verify')
            else: categories.append('Block')
        return categories

def main():
    try:
        detector = FraudDetectionModel()
        logger.info("✅ Model is ready for deployment!")
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")

if __name__ == "__main__":
    main()
