"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'numerical_col': [1.0, 2.0, np.nan, 4.0],
        'categorical_col': ['A', 'B', 'A', 'C'],
        'target': [0, 1, 1, 0]
    })

@pytest.fixture
def preprocessor():
    """Create DataPreprocessor instance."""
    return DataPreprocessor()

def test_preprocess_features(preprocessor, sample_data):
    """Test feature preprocessing."""
    X = preprocessor.preprocess_features(sample_data.drop('target', axis=1))
    
    # Check if missing values are handled
    assert not X.isnull().any().any()
    
    # Check if categorical variables are encoded
    assert X['categorical_col'].dtype in ['int32', 'int64']
    
    # Check if numerical variables are scaled
    assert X['numerical_col'].mean() == pytest.approx(0, abs=1e-10)
    assert X['numerical_col'].std() == pytest.approx(1, abs=1e-10)

def test_split_data(preprocessor, sample_data):
    """Test data splitting."""
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.5)
    
    # Check dimensions
    assert len(X_train) == len(y_train) == 2
    assert len(X_test) == len(y_test) == 2
    
    # Check if data is properly shuffled
    assert not X_train.equals(X.iloc[:2])

def test_save_load_preprocessor(preprocessor, sample_data, tmp_path):
    """Test saving and loading preprocessor objects."""
    # Preprocess data
    X = preprocessor.preprocess_features(sample_data.drop('target', axis=1))
    
    # Save preprocessor
    preprocessor.save_preprocessor(tmp_path)
    
    # Create new preprocessor and load saved objects
    new_preprocessor = DataPreprocessor()
    new_preprocessor.load_preprocessor(tmp_path)
    
    # Check if preprocessor objects are loaded correctly
    assert new_preprocessor.feature_names == preprocessor.feature_names
    assert len(new_preprocessor.label_encoders) == len(preprocessor.label_encoders)
    
    # Process new data and compare results
    new_X = new_preprocessor.preprocess_features(sample_data.drop('target', axis=1))
    pd.testing.assert_frame_equal(X, new_X)