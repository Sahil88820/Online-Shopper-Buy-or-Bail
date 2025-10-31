"""
Smart Shopper AI - FastAPI Backend
RESTful API for predictions, personas, and incentive recommendations
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import uuid

# Initialize FastAPI app
app = FastAPI(
    title="Smart Shopper AI API",
    description="ML-powered shopper behavior prediction and incentive recommendation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Load ML Models
# ============================================================

class ModelLoader:
    """Singleton class to load and cache ML models"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_models()
        return cls._instance
    
    def load_models(self):
        """Load all trained models"""
        try:
            self.purchase_model = joblib.load('models/purchase_predictor_catboost.pkl')
            self.kmeans_model = joblib.load('models/kmeans_model.pkl')
            self.pca_transformer = joblib.load('models/pca_transformer.pkl')
            self.incentive_model = joblib.load('models/incentive_recommender.pkl')
            self.incentive_encoder = joblib.load('models/incentive_label_encoder.pkl')
            self.scaler = joblib.load('data/processed/scaler.pkl')
            self.feature_names = joblib.load('data/processed/feature_names.pkl')
            self.persona_profiles = joblib.load('models/persona_profiles.pkl')
            print("✅ All models loaded successfully!")
        except Exception as e:
            print(f"⚠️  Error loading models: {e}")
            print("Using fallback prediction methods")
            self.purchase_model = None

models = ModelLoader()

# ============================================================
# Pydantic Models (Request/Response schemas)
# ============================================================

class SessionData(BaseModel):
    """Input data for a shopping session"""
    # Original features
    administrative: int = Field(default=0, ge=0)
    administrative_duration: float = Field(default=0.0, ge=0)
    informational: int = Field(default=0, ge=0)
    informational_duration: float = Field(default=0.0, ge=0)
    product_related: int = Field(default=0, ge=0)
    product_related_duration: float = Field(default=0.0, ge=0)
    bounce_rates: float = Field(default=0.0, ge=0, le=1)
    exit_rates: float = Field(default=0.0, ge=0, le=1)
    page_values: float = Field(default=0.0, ge=0)
    special_day: float = Field(default=0.0, ge=0, le=1)
    
    # Categorical (will be encoded)
    month: str = Field(default="Nov")
    visitor_type: str = Field(default="Returning_Visitor")
    weekend: bool = Field(default=False)
    
    operating_systems: int = Field(default=1)
    browser: int = Field(default=1)
    region: int = Field(default=1)
    traffic_type: int = Field(default=1)
    
    # Optional metadata
    device_type: Optional[str] = "desktop"
    user_id: Optional[int] = None

class PredictionResponse(BaseModel):
    """Response for purchase prediction"""
    session_id: str
    timestamp: str
    will_buy: bool
    buy_probability: float
    bail_probability: float
    confidence: float
    model_version: str

class PersonaResponse(BaseModel):
    """Response for persona identification"""
    cluster_id: int
    persona_type: str
    persona_color: str
    size: int
    percentage: float
    characteristics: List[str]

class IncentiveResponse(BaseModel):
    """Response for incentive recommendation"""
    incentive_type: str
    category: str
    message: str
    confidence: float
    discount_percent: Optional[int] = 0
    loyalty_points: Optional[int] = 0
    urgency_level: Optional[str] = None

class CompletePrediction(BaseModel):
    """Complete prediction with all components"""
    session_id: str
    timestamp: str
    prediction: PredictionResponse
    persona: PersonaResponse
    incentive: IncentiveResponse
    feature_importance: Optional[Dict[str, float]] = None

# ============================================================
# Helper Functions
# ============================================================

def engineer_features(data: SessionData) -> Dict:
    """Engineer features from raw session data"""
    # Calculate total duration
    total_duration = (
        data.administrative_duration +
        data.informational_duration +
        data.product_related_duration
    )
    
    # Calculate total pages
    total_pages = (
        data.administrative +
        data.informational +
        data.product_related
    )
    
    # Engagement score
    engagement_score = total_duration / (total_pages + 1)
    
    # Product focus ratio
    product_focus_ratio = data.product_related_duration / (total_duration + 1)
    
    # High bounce/exit flags
    high_bounce = 1 if data.bounce_rates > 0.02 else 0
    high_exit = 1 if data.exit_rates > 0.03 else 0
    
    # Page value per page
    page_value_per_page = data.page_values / (total_pages + 1)
    
    # Session intensity
    session_intensity = total_pages / ((total_duration / 60) + 1)
    
    # Product engagement
    product_engagement = data.product_related / (total_pages + 1)
    
    # Is returning
    is_returning = 1 if data.visitor_type == "Returning_Visitor" else 0
    
    return {
        'TotalDuration': total_duration,
        'TotalPages': total_pages,
        'EngagementScore': engagement_score,
        'ProductFocusRatio': product_focus_ratio,
        'HighBounce': high_bounce,
        'HighExit': high_exit,
        'PageValuePerPage': page_value_per_page,
        'SessionIntensity': session_intensity,
        'ProductEngagement': product_engagement,
        'IsReturning': is_returning
    }

def prepare_features(data: SessionData) -> pd.DataFrame:
    """Prepare features for model prediction"""
    # Encode categorical variables
    month_map = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
                 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
    visitor_map = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 2}
    
    # Engineer features
    engineered = engineer_features(data)
    
    # Combine all features
    features = {
        'Administrative': data.administrative,
        'Administrative_Duration': data.administrative_duration,
        'Informational': data.informational,
        'Informational_Duration': data.informational_duration,
        'ProductRelated': data.product_related,
        'ProductRelated_Duration': data.product_related_duration,
        'BounceRates': data.bounce_rates,
        'ExitRates': data.exit_rates,
        'PageValues': data.page_values,
        'SpecialDay': data.special_day,
        'Month_Encoded': month_map.get(data.month, 10),
        'VisitorType_Encoded': visitor_map.get(data.visitor_type, 1),
        'Weekend_Encoded': int(data.weekend),
        'OperatingSystems': data.operating_systems,
        'Browser': data.browser,
        'Region': data.region,
        'TrafficType': data.traffic_type,
        **engineered
    }
    
    # Create DataFrame with correct column order
    df = pd.DataFrame([features])
    
    # Ensure all required features are present
    for feature in models.feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    return df[models.feature_names]

# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Smart Shopper AI API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "personas": "/personas",
            "analytics": "/analytics",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": models.purchase_model is not None
    }

@app.post("/predict", response_model=CompletePrediction)
async def predict(session: SessionData):
    """
    Make complete prediction for a shopping session
    Includes: purchase prediction, persona identification, and incentive recommendation
    """
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Prepare features
        X = prepare_features(session)
        
        # 1. Purchase Prediction
        if models.purchase_model:
            buy_proba = models.purchase_model.predict_proba(X)[0, 1]
            will_buy = bool(buy_proba > 0.5)
        else:
            # Fallback prediction
            buy_proba = 0.5
            will_buy = False
        
        prediction = PredictionResponse(
            session_id=session_id,
            timestamp=timestamp,
            will_buy=will_buy,
            buy_probability=float(buy_proba),
            bail_probability=float(1 - buy_proba),
            confidence=float(max(buy_proba, 1 - buy_proba) * 100),
            model_version="v1.0"
        )
        
        # 2. Persona Identification
        if models.kmeans_model:
            cluster_id = int(models.kmeans_model.predict(X)[0])
            persona_profile = models.persona_profiles.get(cluster_id, {})
            
            persona = PersonaResponse(
                cluster_id=cluster_id,
                persona_type=persona_profile.get('name', f'Cluster {cluster_id}'),
                persona_color=["purple", "green", "blue", "yellow", "orange"][cluster_id],
                size=persona_profile.get('size', 0),
                percentage=persona_profile.get('percentage', 0.0),
                characteristics=persona_profile.get('characteristics', [])
            )
        else:
            persona = PersonaResponse(
                cluster_id=0,
                persona_type="Unknown",
                persona_color="gray",
                size=0,
                percentage=0.0,
                characteristics=[]
            )
        
        # 3. Incentive Recommendation
        if models.incentive_model:
            # Add cluster and probability features
            X_incentive = X.copy()
            X_incentive['cluster'] = cluster_id
            X_incentive['buy_probability'] = buy_proba
            X_incentive['bail_probability'] = 1 - buy_proba
            
            incentive_pred = models.incentive_model.predict(X_incentive)[0]
            incentive_type = models.incentive_encoder.inverse_transform([incentive_pred])[0]
            incentive_probs = models.incentive_model.predict_proba(X_incentive)[0]
            confidence = float(max(incentive_probs) * 100)
        else:
            incentive_type = 'discount_10'
            confidence = 50.0
        
        # Map incentive to details
        incentive_details = get_incentive_details(incentive_type)
        
        incentive = IncentiveResponse(
            incentive_type=incentive_type,
            confidence=confidence,
            **incentive_details
        )
        
        # 4. Feature Importance (top 5)
        if models.purchase_model and hasattr(models.purchase_model, 'get_feature_importance'):
            importance = models.purchase_model.get_feature_importance()
            top_5_idx = np.argsort(importance)[-5:]
            feature_importance = {
                models.feature_names[i]: float(importance[i])
                for i in top_5_idx
            }
        else:
            feature_importance = None
        
        return CompletePrediction(
            session_id=session_id,
            timestamp=timestamp,
            prediction=prediction,
            persona=persona,
            incentive=incentive,
            feature_importance=feature_importance
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/personas")
async def get_personas():
    """Get all persona profiles"""
    if models.persona_profiles:
        return {
            "personas": [
                {
                    "cluster_id": cid,
                    **profile
                }
                for cid, profile in models.persona_profiles.items()
            ]
        }
    else:
        raise HTTPException(status_code=503, detail="Persona profiles not loaded")

@app.get("/analytics")
async def get_analytics():
    """Get analytics summary"""
    # In production, this would query the database
    return {
        "total_sessions": 1523,
        "conversion_rate": 0.67,
        "avg_session_time": 245.3,
        "top_persona": "Research Shopper",
        "incentives_shown": 842,
        "incentives_accepted": 623,
        "acceptance_rate": 0.74,
        "period": "Last 7 days"
    }

def get_incentive_details(incentive_type: str) -> Dict:
    """Get detailed message and parameters for incentive"""
    incentive_map = {
        'discount_10': {
            'category': 'discount',
            'message': '🎉 Get 10% OFF your order - Limited time!',
            'discount_percent': 10
        },
        'discount_15': {
            'category': 'discount',
            'message': '💰 Save 15% on your cart - Don\'t miss out!',
            'discount_percent': 15
        },
        'discount_20': {
            'category': 'discount',
            'message': '🔥 SPECIAL: 20% OFF - Expires soon!',
            'discount_percent': 20
        },
        'loyalty_points': {
            'category': 'loyalty',
            'message': '⭐ Earn 500 bonus loyalty points on this purchase!',
            'loyalty_points': 500
        },
        'free_shipping': {
            'category': 'shipping',
            'message': '🚚 FREE Shipping on your order - Today only!'
        },
        'urgency_banner': {
            'category': 'urgency',
            'message': '⏰ Only 2 left in stock! Order in next 10 minutes',
            'urgency_level': 'high'
        },
        'none': {
            'category': 'none',
            'message': '✅ Great choice! Proceeding to checkout...'
        }
    }
    
    return incentive_map.get(incentive_type, incentive_map['none'])

# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
