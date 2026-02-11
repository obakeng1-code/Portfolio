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

# --- PATH CONFIGURATION ---
# This approach works better for Streamlit Cloud deployment
BASE_DIR = Path(__file__).resolve().parent
# --------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDetectionModel:
    def __init__(self, model_name='fraud_model.pkl', scaler_name='scaler.pkl', features_name='feature_order.pkl'):
        # Construct absolute paths
        self.model_path = BASE_DIR / model_name
        self.scaler_path = BASE_DIR / scaler_name
        self.features_path = BASE_DIR / features_name
        
        self.model = None
        self.scaler = None
        self.feature_order = None
        
        self.load_artifacts()

    def load_artifacts(self):
        """Load model, scaler, and feature order with detailed error reporting."""
        try:
            # 1. Load Model
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"✅ Model loaded from {self.model_path}")
            else:
                logger.error(f"❌ Model file not found at {self.model_path}")
                raise FileNotFoundError(f"Missing {self.model_path}")

            # 2. Load Scaler
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✅ Scaler loaded from {self.scaler_path}")
            else:
                # Fallback: check current working directory
                cwd_path = Path.cwd() / "scaler.pkl"
                if cwd_path.exists():
                    self.scaler = joblib.load(cwd_path)
                    logger.info(f"✅ Scaler loaded from fallback CWD: {cwd_path}")
                else:
                    logger.error(f"❌ Scaler file not found at {self.scaler_path}")
                    raise FileNotFoundError(f"Missing {self.scaler_path}")

            # 3. Load Feature Order
            if self.features_path.exists():
                self.feature_order = joblib.load(self.features_path)
                logger.info(f"✅ Feature order loaded from {self.features_path}")
            else:
                logger.error(f"❌ Feature order file not found at {self.features_path}")
                raise FileNotFoundError(f"Missing {self.features_path}")
                
        except Exception as e:
            logger.error(f"❌ Critical error loading artifacts: {e}")
            raise

    def preprocess(self, data):
        """Fixes column mismatches and applies scaling."""
        df = data.copy()

        # Remove targets if present
        if 'Class' in df.columns:
            df = df.drop(columns=['Class'])

        # Rename to match training features
        rename_map = {'Amount': 'scaled_amount', 'Time': 'scaled_time'}
        df = df.rename(columns=rename_map)

        # Apply scaler to raw columns
        try:
            cols_to_scale = ['scaled_amount', 'scaled_time']
            if all(col in df.columns for col in cols_to_scale):
                df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        except Exception as e:
            logger.warning(f"⚠️ Scaling application failed: {e}")

        # Ensure all features exist
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0.0

        # Reorder columns
        return df[self.feature_order]

    def predict(self, data):
        """Returns fraud probability (0.0 to 1.0)."""
        processed_data = self.preprocess(data)
        return self.model.predict_proba(processed_data)[:, 1]

    def predict_batch(self, data, threshold=0.5):
        """Returns a DataFrame with results."""
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
        """Categorizes scores into Risk Levels."""
        categories = []
        for score in scores:
            if score < 30: categories.append('Auto-Approve')
            elif score < 85: categories.append('Verify')
            else: categories.append('Block')
        return categories

def main():
    try:
        detector = FraudDetectionModel()
        logger.info("🚀 System ready for inference.")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")

if __name__ == "__main__":
    main()
