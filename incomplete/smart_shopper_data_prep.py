"""
Smart Shopper AI - Data Preprocessing Pipeline
Loads, cleans, and engineers features from the Online Shoppers Intention Dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ShopperDataPreprocessor:
    """Handles data loading, cleaning, and feature engineering"""
    
    def __init__(self, data_path='online_shoppers_intention.csv'):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self):
        """Load the Online Shoppers Intention dataset"""
        print("Loading Online Shoppers Intention Dataset...")
        df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def clean_data(self, df):
        """Clean and handle missing values"""
        print("\nCleaning data...")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"Missing values found:\n{missing[missing > 0]}")
            df = df.dropna()
        
        # Remove duplicates
        before = len(df)
        df = df.drop_duplicates()
        print(f"Removed {before - len(df)} duplicate rows")
        
        # Remove outliers using IQR method for key numeric features
        numeric_cols = ['ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            before = len(df)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        print(f"Dataset after cleaning: {df.shape[0]} rows")
        return df
    
    def engineer_features(self, df):
        """Create new features from existing data"""
        print("\nEngineering features...")
        
        # 1. Total time on site
        df['TotalDuration'] = (
            df['Administrative_Duration'] + 
            df['Informational_Duration'] + 
            df['ProductRelated_Duration']
        )
        
        # 2. Total pages visited
        df['TotalPages'] = (
            df['Administrative'] + 
            df['Informational'] + 
            df['ProductRelated']
        )
        
        # 3. Engagement score (time per page)
        df['EngagementScore'] = df['TotalDuration'] / (df['TotalPages'] + 1)
        
        # 4. Product focus ratio
        df['ProductFocusRatio'] = (
            df['ProductRelated_Duration'] / (df['TotalDuration'] + 1)
        )
        
        # 5. High bounce flag
        df['HighBounce'] = (df['BounceRates'] > df['BounceRates'].median()).astype(int)
        
        # 6. High exit flag
        df['HighExit'] = (df['ExitRates'] > df['ExitRates'].median()).astype(int)
        
        # 7. Weekend shopper flag
        df['IsWeekend'] = df['Weekend'].astype(int)
        
        # 8. Special day proximity score
        df['SpecialDayProximity'] = df['SpecialDay']
        
        # 9. Page value per page
        df['PageValuePerPage'] = df['PageValues'] / (df['TotalPages'] + 1)
        
        # 10. Returning visitor flag
        df['IsReturning'] = (df['VisitorType'] == 'Returning_Visitor').astype(int)
        
        # 11. Session intensity (pages per minute)
        df['SessionIntensity'] = df['TotalPages'] / ((df['TotalDuration'] / 60) + 1)
        
        # 12. Product engagement (product pages per total pages)
        df['ProductEngagement'] = df['ProductRelated'] / (df['TotalPages'] + 1)
        
        print(f"Created {12} new features")
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        print("\nEncoding categorical variables...")
        
        categorical_cols = ['Month', 'VisitorType', 'Weekend']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_Encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                print(f"  {col}: {len(le.classes_)} categories")
        
        return df
    
    def select_features(self, df):
        """Select and prepare final feature set"""
        print("\nSelecting features for modeling...")
        
        # Features for prediction
        feature_cols = [
            # Original features
            'Administrative', 'Administrative_Duration',
            'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration',
            'BounceRates', 'ExitRates', 'PageValues',
            'SpecialDay',
            'Month_Encoded', 'VisitorType_Encoded', 'Weekend_Encoded',
            'OperatingSystems', 'Browser', 'Region', 'TrafficType',
            
            # Engineered features
            'TotalDuration', 'TotalPages', 'EngagementScore',
            'ProductFocusRatio', 'HighBounce', 'HighExit',
            'PageValuePerPage', 'IsReturning', 'SessionIntensity',
            'ProductEngagement'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_names = available_features
        
        X = df[available_features]
        y = df['Revenue'].astype(int)
        
        print(f"Selected {len(available_features)} features")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features"""
        print("\nScaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Complete preprocessing pipeline"""
        print("="*60)
        print("SMART SHOPPER AI - DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load data
        df = self.load_data()
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode categorical
        df = self.encode_categorical(df)
        
        # Select features
        X, y = self.select_features(df)
        
        # Split data
        print(f"\nSplitting data (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Feature names: {len(self.feature_names)}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'raw_data': df
        }
    
    def save_processed_data(self, data_dict, output_dir='data/processed'):
        """Save processed data for later use"""
        import os
        import joblib
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save datasets
        data_dict['X_train'].to_csv(f'{output_dir}/X_train.csv', index=False)
        data_dict['X_test'].to_csv(f'{output_dir}/X_test.csv', index=False)
        data_dict['y_train'].to_csv(f'{output_dir}/y_train.csv', index=False)
        data_dict['y_test'].to_csv(f'{output_dir}/y_test.csv', index=False)
        
        # Save preprocessors
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{output_dir}/label_encoders.pkl')
        joblib.dump(self.feature_names, f'{output_dir}/feature_names.pkl')
        
        print(f"\nProcessed data saved to {output_dir}/")


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ShopperDataPreprocessor('online_shoppers_intention.csv')
    
    # Run preprocessing pipeline
    data = preprocessor.prepare_data(test_size=0.2, random_state=42)
    
    # Save processed data
    preprocessor.save_processed_data(data)
    
    print("\n✅ Data preprocessing completed successfully!")
    print(f"   Features: {len(data['feature_names'])}")
    print(f"   Training samples: {len(data['X_train'])}")
    print(f"   Test samples: {len(data['X_test'])}")
