"""
FastAPI backend for Smart Shopper AI.
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
import numpy as np
import pandas as pd

from src.database import get_db, Shopper, Prediction
from src.purchase_prediction import PurchasePredictor
from src.persona_clustering import PersonaClusterer
from src.incentive_recommendation import IncentiveRecommender
from src.data_preprocessing import DataPreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="Smart Shopper AI API",
    description="API for predicting shopper behavior and recommending incentives",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
predictor = PurchasePredictor()
predictor.load_model("models")

clusterer = PersonaClusterer()
clusterer.load_model("models")

recommender = IncentiveRecommender()
recommender.load_model("models")

preprocessor = DataPreprocessor()
preprocessor.load_preprocessor("data/processed")

# Pydantic models
class ShopperBase(BaseModel):
    session_id: str
    administrative_pages: int
    informational_pages: int
    product_pages: int
    bounce_rates: float
    exit_rates: float
    page_values: float
    special_day: float
    month: str
    weekend: bool
    operating_system: str
    browser: str
    region: str
    traffic_type: int
    visitor_type: str

class PredictionResponse(BaseModel):
    purchase_probability: float
    persona_id: int
    recommended_incentive: str
    persona_description: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_shopper(shopper: ShopperBase, db: Session = Depends(get_db)):
    """Generate predictions and recommendations for a shopper."""
    try:
        # Convert shopper data to DataFrame
        shopper_dict = shopper.dict()
        df = pd.DataFrame([shopper_dict])
        
        # Preprocess features
        X = preprocessor.preprocess_features(df)
        
        # Generate predictions
        purchase_prob = predictor.predict_proba(X)[0][1]
        persona = clusterer.predict(X)[0]
        incentive = recommender.predict(X)[0]
        
        # Get persona description
        persona_desc = clusterer.persona_profiles[persona]['description']
        
        # Save to database
        db_shopper = Shopper(**shopper_dict)
        db.add(db_shopper)
        db.commit()
        
        db_prediction = Prediction(
            shopper_id=db_shopper.id,
            purchase_probability=float(purchase_prob),
            predicted_persona=int(persona),
            recommended_incentive=incentive
        )
        db.add(db_prediction)
        db.commit()
        
        return PredictionResponse(
            purchase_probability=purchase_prob,
            persona_id=persona,
            recommended_incentive=incentive,
            persona_description=persona_desc
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check API health status."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)