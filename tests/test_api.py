"""
Integration tests for FastAPI backend.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

from src.api import app

client = TestClient(app)

@pytest.fixture
def sample_shopper():
    """Create sample shopper data."""
    return {
        "session_id": "test_session_001",
        "administrative_pages": 2,
        "informational_pages": 3,
        "product_pages": 4,
        "bounce_rates": 0.2,
        "exit_rates": 0.15,
        "page_values": 10.5,
        "special_day": 0.0,
        "month": "May",
        "weekend": True,
        "operating_system": "Windows",
        "browser": "Chrome",
        "region": "North America",
        "traffic_type": 1,
        "visitor_type": "New"
    }

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "timestamp" in response.json()

def test_predict_endpoint(sample_shopper):
    """Test prediction endpoint."""
    response = client.post("/predict", json=sample_shopper)
    assert response.status_code == 200
    
    data = response.json()
    assert "purchase_probability" in data
    assert "persona_id" in data
    assert "recommended_incentive" in data
    assert "persona_description" in data
    
    assert 0 <= data["purchase_probability"] <= 1
    assert isinstance(data["persona_id"], int)
    assert isinstance(data["recommended_incentive"], str)
    assert isinstance(data["persona_description"], str)

def test_predict_invalid_data():
    """Test prediction endpoint with invalid data."""
    invalid_shopper = {
        "session_id": "test_invalid"
        # Missing required fields
    }
    
    response = client.post("/predict", json=invalid_shopper)
    assert response.status_code == 422  # Validation error

def test_predict_missing_features(sample_shopper):
    """Test prediction endpoint with missing features."""
    # Remove a required feature
    del sample_shopper["bounce_rates"]
    
    response = client.post("/predict", json=sample_shopper)
    assert response.status_code == 422

def test_predict_invalid_values(sample_shopper):
    """Test prediction endpoint with invalid values."""
    # Set invalid value
    sample_shopper["bounce_rates"] = -1
    
    response = client.post("/predict", json=sample_shopper)
    assert response.status_code == 400  # Bad request