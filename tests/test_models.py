"""
Unit tests for machine learning models.
"""

import pytest
import pandas as pd
import numpy as np
from src.purchase_prediction import PurchasePredictor
from src.persona_clustering import PersonaClusterer
from src.incentive_recommendation import IncentiveRecommender

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })

def test_purchase_predictor(sample_data, tmp_path):
    """Test purchase prediction model."""
    # Prepare data
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']
    
    # Initialize and train model
    predictor = PurchasePredictor()
    predictor.fit(X, y)
    
    # Test predictions
    predictions = predictor.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})
    
    # Test probability predictions
    probabilities = predictor.predict_proba(X)
    assert probabilities.shape == (len(y), 2)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    
    # Test feature importance
    assert predictor.feature_importance is not None
    assert len(predictor.feature_importance) == X.shape[1]
    
    # Test save/load
    predictor.save_model(tmp_path)
    new_predictor = PurchasePredictor()
    new_predictor.load_model(tmp_path)
    
    np.testing.assert_array_almost_equal(
        predictor.predict_proba(X),
        new_predictor.predict_proba(X)
    )

def test_persona_clusterer(sample_data, tmp_path):
    """Test persona clustering model."""
    # Prepare data
    X = sample_data.drop('target', axis=1)
    
    # Initialize and fit model
    clusterer = PersonaClusterer(n_clusters=3)
    clusterer.fit(X)
    
    # Test predictions
    clusters = clusterer.predict(X)
    assert len(clusters) == len(X)
    assert set(clusters).issubset({0, 1, 2})
    
    # Test PCA transformation
    pca_coords = clusterer.transform_pca(X)
    assert pca_coords.shape == (len(X), 2)
    
    # Test persona profiles
    assert len(clusterer.persona_profiles) == 3
    for profile in clusterer.persona_profiles.values():
        assert 'description' in profile
        
    # Test save/load
    clusterer.save_model(tmp_path)
    new_clusterer = PersonaClusterer(n_clusters=3)
    new_clusterer.load_model(tmp_path)
    
    np.testing.assert_array_equal(
        clusterer.predict(X),
        new_clusterer.predict(X)
    )

def test_incentive_recommender(sample_data, tmp_path):
    """Test incentive recommendation model."""
    # Prepare data
    X = sample_data.drop('target', axis=1)
    incentives = ['discount_10', 'discount_20', 'free_shipping']
    y = np.random.choice(incentives, len(X))
    
    # Initialize and train model
    recommender = IncentiveRecommender()
    recommender.fit(X, y)
    
    # Test predictions
    predictions = recommender.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions).issubset(set(incentives))
    
    # Test probability predictions
    probabilities = recommender.predict_proba(X)
    assert probabilities.shape == (len(y), len(incentives))
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    
    # Test incentive types
    assert set(recommender.get_incentive_types()) == set(incentives)
    
    # Test save/load
    recommender.save_model(tmp_path)
    new_recommender = IncentiveRecommender()
    new_recommender.load_model(tmp_path)
    
    np.testing.assert_array_equal(
        recommender.predict(X),
        new_recommender.predict(X)
    )