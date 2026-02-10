"""
Credit Card Fraud Detection - Main Inference Script
Loads trained models and provides fraud prediction capabilities
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """
    Wrapper class for loading and using the trained fraud detection model.
    """
    
    def __init__(self, model_path='fraud_model.pkl', scaler_path='scaler.pkl', 
                 features_path='feature_order.pkl'):
        """
        Initialize the model loader.
        
        Args:
            model_path (str): Path to the trained model pickle file
            scaler_path (str): Path to the scaler pickle file
            features_path (str): Path to the feature order pickle file
        """
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
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✅ Scaler loaded from {self.scaler_path}")
            else:
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            
            if self.features_path.exists():
                self.feature_order = joblib.load(self.features_path)
                logger.info(f"✅ Feature order loaded from {self.features_path}")
            else:
                raise FileNotFoundError(f"Feature order file not found: {self.features_path}")
                
        except Exception as e:
            logger.error(f"❌ Error loading artifacts: {e}")
            raise
    
    def preprocess(self, data):
        """
        Preprocess input data to match training format.
        
        Args:
            data (pd.DataFrame): Input transaction data
            
        Returns:
            pd.DataFrame: Preprocessed data ready for prediction
        """
        logger.info(f"Expected {len(self.feature_order)} features, got {len(data.columns)} in input")
        
        # Ensure columns are in the correct order
        if not all(col in data.columns for col in self.feature_order):
            missing = set(self.feature_order) - set(data.columns)
            extra = set(data.columns) - set(self.feature_order)
            error_msg = f"\nFeature mismatch:\n"
            if missing:
                error_msg += f"  Missing: {missing}\n"
            if extra:
                error_msg += f"  Extra (not needed): {extra}\n"
            error_msg += f"\nExpected features ({len(self.feature_order)}): {self.feature_order}\n"
            error_msg += f"Input features ({len(data.columns)}): {list(data.columns)}"
            raise ValueError(error_msg)
        
        # Ensure we have exactly the right number of features
        if len(data.columns) != len(self.feature_order):
            logger.warning(f"Column count mismatch: expected {len(self.feature_order)}, got {len(data.columns)}")
        
        # Reorder columns
        data = data[self.feature_order].copy()
        
        return data
    
    def predict(self, data):
        """
        Predict fraud probability for input transactions.
        
        Args:
            data (pd.DataFrame): Transaction data with required features
            
        Returns:
            np.ndarray: Fraud probability scores (0-1)
        """
        try:
            # Preprocess
            processed = self.preprocess(data)
            
            # Get probability predictions
            probs = self.model.predict_proba(processed)[:, 1]
            
            return probs
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            raise
    
    def predict_batch(self, data, threshold=0.5):
        """
        Predict fraud for a batch of transactions with decision threshold.
        
        Args:
            data (pd.DataFrame): Batch of transactions
            threshold (float): Classification threshold (0-1)
            
        Returns:
            dict: Predictions with scores and classifications
        """
        try:
            probs = self.predict(data)
            predictions = (probs >= threshold).astype(int)
            
            result = pd.DataFrame({
                'Risk_Score': (probs * 100).round(2),
                'Prediction': predictions,
                'Prediction_Label': ['Fraud' if p == 1 else 'Normal' for p in predictions]
            })
            
            return result
        except Exception as e:
            logger.error(f"❌ Batch prediction error: {e}")
            raise
    
    def categorize_risk(self, scores):
        """
        Categorize risk scores into business action categories.
        
        Args:
            scores (np.ndarray or list): Risk scores (0-100)
            
        Returns:
            list: Risk categories
        """
        categories = []
        for score in scores:
            if score < 30:
                categories.append('Auto-Approve')
            elif score < 85:
                categories.append('Verify')
            else:
                categories.append('Block')
        return categories


def main():
    """Example usage of the fraud detection model."""
    
    try:
        # Initialize model
        detector = FraudDetectionModel()
        
        # Example: Single transaction prediction
        logger.info("\n" + "="*60)
        logger.info("FRAUD DETECTION MODEL - EXAMPLE USAGE")
        logger.info("="*60)
        
        # Create dummy transaction (you would replace with real data)
        dummy_transaction = pd.DataFrame({
            col: [0.0] for col in detector.feature_order
        })
        
        logger.info("\n1. Single Transaction Prediction:")
        prob = detector.predict(dummy_transaction)[0]
        risk_score = prob * 100
        logger.info(f"   Risk Score: {risk_score:.2f}%")
        logger.info(f"   Decision: {detector.categorize_risk([risk_score])[0]}")
        
        # Example: Batch prediction
        logger.info("\n2. Batch Predictions:")
        batch_data = pd.concat([dummy_transaction] * 5, ignore_index=True)
        predictions = detector.predict_batch(batch_data)
        logger.info(f"\n{predictions.head()}")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Model is ready for deployment!")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        raise


if __name__ == "__main__":
    main()