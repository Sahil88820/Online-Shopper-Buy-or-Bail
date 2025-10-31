"""
Data preprocessing module for Smart Shopper AI.
Handles data cleaning, feature engineering, and transformation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load raw data from CSV file."""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded data from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_features(self, df):
        """Clean and transform features."""
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
            
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        self.feature_names = df.columns.tolist()
        return df
        
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    def save_preprocessor(self, output_dir):
        """Save preprocessing objects."""
        with open(f"{output_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f"{output_dir}/label_encoders.pkl", 'wb') as f:
            pickle.dump(self.label_encoders, f)
        with open(f"{output_dir}/feature_names.pkl", 'wb') as f:
            pickle.dump(self.feature_names, f)
            
    def load_preprocessor(self, input_dir):
        """Load preprocessing objects."""
        with open(f"{input_dir}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        with open(f"{input_dir}/label_encoders.pkl", 'rb') as f:
            self.label_encoders = pickle.load(f)
        with open(f"{input_dir}/feature_names.pkl", 'rb') as f:
            self.feature_names = pickle.load(f)

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    df = preprocessor.load_data("data/raw/online_shoppers_intention.csv")
    X = preprocessor.preprocess_features(df.drop('Revenue', axis=1))
    y = df['Revenue']
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Save processed data
    pd.DataFrame(X_train, columns=preprocessor.feature_names).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test, columns=preprocessor.feature_names).to_csv("data/processed/X_test.csv", index=False)
    pd.Series(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.Series(y_test).to_csv("data/processed/y_test.csv", index=False)
    
    # Save preprocessor objects
    preprocessor.save_preprocessor("data/processed")