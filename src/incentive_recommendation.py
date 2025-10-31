"""
Incentive recommendation module using XGBoost multi-class classifier.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncentiveRecommender:
    def __init__(self, random_state=42):
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state
        )
        self.label_encoder = LabelEncoder()
        
    def fit(self, X, y):
        """Train the XGBoost model."""
        try:
            # Encode incentive types
            y_encoded = self.label_encoder.fit_transform(y)
            self.model.fit(X, y_encoded)
            logger.info("Successfully trained XGBoost model")
            return self
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def predict(self, X):
        """Predict incentive types."""
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)
        
    def predict_proba(self, X):
        """Get probability distribution across incentive types."""
        return self.model.predict_proba(X)
        
    def get_incentive_types(self):
        """Get list of possible incentive types."""
        return self.label_encoder.classes_
        
    def save_model(self, output_dir):
        """Save model and label encoder."""
        with open(f"{output_dir}/incentive_recommender.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        with open(f"{output_dir}/incentive_label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
    def load_model(self, input_dir):
        """Load model and label encoder."""
        with open(f"{input_dir}/incentive_recommender.pkl", 'rb') as f:
            self.model = pickle.load(f)
        with open(f"{input_dir}/incentive_label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)

if __name__ == "__main__":
    # Example usage
    # Load preprocessed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    
    # Create synthetic incentive data for demonstration
    incentive_types = [
        'discount_10', 'discount_20', 'free_shipping',
        'bundle_deal', 'loyalty_points', 'flash_sale',
        'gift_card'
    ]
    y_train = np.random.choice(incentive_types, size=len(X_train))
    
    # Create and train recommender
    recommender = IncentiveRecommender()
    recommender.fit(X_train, y_train)
    
    # Save model
    recommender.save_model("models")