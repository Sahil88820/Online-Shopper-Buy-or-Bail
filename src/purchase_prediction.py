"""
Purchase prediction module using CatBoost classifier.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import shap
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PurchasePredictor:
    def __init__(self, random_state=42):
        self.model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            random_state=random_state,
            verbose=100
        )
        self.feature_importance = None
        
    def fit(self, X, y):
        """Train the CatBoost model."""
        try:
            self.model.fit(X, y)
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info("Successfully trained CatBoost model")
            return self
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def predict(self, X):
        """Make binary predictions."""
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Get probability predictions."""
        return self.model.predict_proba(X)
        
    def explain_predictions(self, X):
        """Generate SHAP values for model explanations."""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return shap_values
        
    def save_model(self, output_dir):
        """Save model and feature importance."""
        with open(f"{output_dir}/purchase_predictor_catboost.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        self.feature_importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)
            
    def load_model(self, input_dir):
        """Load model and feature importance."""
        with open(f"{input_dir}/purchase_predictor_catboost.pkl", 'rb') as f:
            self.model = pickle.load(f)
        self.feature_importance = pd.read_csv(f"{input_dir}/feature_importance.csv")

if __name__ == "__main__":
    # Example usage
    # Load preprocessed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").iloc[:, 0]
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]
    
    # Create and train predictor
    predictor = PurchasePredictor()
    predictor.fit(X_train, y_train)
    
    # Save model
    predictor.save_model("models")