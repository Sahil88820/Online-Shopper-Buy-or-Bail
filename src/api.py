"""
FastAPI backend for Smart Shopper AI

Endpoints:
- POST /predict : run full prediction (purchase + persona + incentive)
- GET /personas : return persona profiles
- GET /health   : quick check

Run:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""
from sqlalchemy.orm import Session
from src.database import (
    get_db, Session as DBSession, Prediction as DBPrediction,
    Persona as DBPersona, Incentive as DBIncentive
)

import uuid
import json
import pickle
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends
from pydantic import BaseModel

# ---------------------------------------------------------------------
# MODELS & PREPROCESSORS PATHS
# ---------------------------------------------------------------------
BASE = "./"
MODEL_DIR = f"{BASE}/models"
DATA_DIR = f"{BASE}/data/processed"


# print("⚠️ Model loading disabled for debug")
# purchase_model = None

# Load models once on startup (cached in memory)
print("🚀 Loading models & preprocessors...")

purchase_model = pickle.load(open(f"{MODEL_DIR}/purchase_predictor_catboost.pkl", "rb"))
kmeans_model = pickle.load(open(f"{MODEL_DIR}/kmeans_model.pkl", "rb"))
pca_model = pickle.load(open(f"{MODEL_DIR}/pca_transformer.pkl", "rb"))
persona_profiles = pickle.load(open(f"{MODEL_DIR}/persona_profiles.pkl", "rb"))
incentive_model = pickle.load(open(f"{MODEL_DIR}/incentive_recommender.pkl", "rb"))
incentive_encoder = pickle.load(open(f"{MODEL_DIR}/incentive_label_encoder.pkl", "rb"))

scaler = pickle.load(open(f"{DATA_DIR}/scaler.pkl", "rb"))
label_encoders = pickle.load(open(f"{DATA_DIR}/label_encoders.pkl", "rb"))
feature_names = pickle.load(open(f"{DATA_DIR}/feature_names.pkl", "rb"))

print("✅ Models loaded successfully!")

# ---------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------
app = FastAPI(
    title="Smart Shopper AI",
    description="Predict shopper behavior + persona + incentive",
    version="1.0.0"
)

# ---------------------------------------------------------------------
# REQUEST PAYLOAD
# ---------------------------------------------------------------------
class SessionInput(BaseModel):
    administrative: float
    administrative_duration: float
    informational: float
    informational_duration: float
    product_related: float
    product_related_duration: float
    bounce_rates: float
    exit_rates: float
    page_values: float
    special_day: float
    month: str
    visitor_type: str
    weekend: bool
    operating_systems: int
    browser: int
    region: int
    traffic_type: int

# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df["TotalDuration"] = (
        df["Administrative_Duration"] +
        df["Informational_Duration"] +
        df["ProductRelated_Duration"]
    )

    df["TotalPages"] = (
        df["Administrative"] +
        df["Informational"] +
        df["ProductRelated"]
    )
    df["EngagementScore"] = df["TotalDuration"] * (1 - df["BounceRates"])
    df["ProductFocusRatio"] = df["ProductRelated"] / (df["TotalPages"] + 1)
    df["SessionIntensity"] = df["TotalDuration"] / (df["TotalPages"] + 1)
    df["ExitFlag"] = (df["ExitRates"] > 0.1).astype(int)
    df["BounceFlag"] = (df["BounceRates"] > 0.1).astype(int)
    return df

def preprocess(input_dict: dict) -> np.ndarray:
    # Map API fields to training feature names
    rename_map = {
        "administrative": "Administrative",
        "administrative_duration": "Administrative_Duration",
        "informational": "Informational",
        "informational_duration": "Informational_Duration",
        "product_related": "ProductRelated",
        "product_related_duration": "ProductRelated_Duration",
        "bounce_rates": "BounceRates",
        "exit_rates": "ExitRates",
        "page_values": "PageValues",
        "special_day": "SpecialDay",
        "month": "Month",
        "visitor_type": "VisitorType",
        "weekend": "Weekend",
        "operating_systems": "OperatingSystems",
        "browser": "Browser",
        "region": "Region",
        "traffic_type": "TrafficType"
    }

    input_dict = {rename_map[k]: v for k, v in input_dict.items()}
    df = pd.DataFrame([input_dict])

    # ✅ Accept only short month names (because encoder expects them)
    allowed_months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    if "Month" in df.columns:
        m = df["Month"].iloc[0]
        print("🗓 Month provided:", m)

        if m not in allowed_months:
            raise ValueError(f"❌ Invalid month '{m}'. Use: {allowed_months}")

        df["Month"] = m.strip()

    # ✅ Normalise visitor type
    visitor_map = {
        "Returning Visitor": "Returning_Visitor",
        "New Visitor": "New_Visitor",
        "Returning_Visitor": "Returning_Visitor",
        "New_Visitor": "New_Visitor"
    }

    if "VisitorType" in df.columns:
        df["VisitorType"] = df["VisitorType"].map(visitor_map).fillna(df["VisitorType"])

    # ✅ Label encoding (month will stay as string here)
    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])

    print("✅ Encoded sample:", df[["Month", "VisitorType"]])
    print(label_encoders["Month"].classes_)
    # ✅ Feature engineering and scaling
    df = feature_engineer(df)
    df = df[feature_names]
    return scaler.transform(df)


def choose_incentive(prob, persona):
    """
    Simple rule-layer on top of model prediction
    """

    if prob > 0.7:
        return "none"

    # Boost discount for Deal Hunters
    if persona.lower() == "deal hunter":
        return "discount_20"

    return None  # let ML model decide


# ---------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": True,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/personas")
def personas():
    return {"personas": persona_profiles}

@app.post("/predict")
def predict(input_data: SessionInput, db: Session = Depends(get_db)):
    # Convert request to dict
    input_dict = input_data.dict()
    print("📥 Received input:", input_dict)


    # ✅ Generate session ID and timestamp
    session_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()

    # ✅ Convert input to model features
    X_processed = preprocess(input_dict)

    # ✅ Main model prediction
    buy_prob = float(purchase_model.predict_proba(X_processed)[0][1])
    will_buy = buy_prob >= 0.5

    # ✅ Persona prediction
    cluster_id = int(kmeans_model.predict(X_processed)[0])
    persona = persona_profiles[cluster_id]["persona_type"]

    # ✅ Incentive prediction
    incentive_pred = incentive_model.predict(X_processed)
    incentive = incentive_encoder.inverse_transform(incentive_pred)[0]
    confidence = float(np.max(incentive_model.predict_proba(X_processed)))

    # ---------------- DB SAVE ----------------
    db_session = DBSession(
        session_id=session_id,
        administrative=input_dict["administrative"],
        administrative_duration=input_dict["administrative_duration"],
        informational=input_dict["informational"],
        informational_duration=input_dict["informational_duration"],
        product_related=input_dict["product_related"],
        product_related_duration=input_dict["product_related_duration"],
        bounce_rates=input_dict["bounce_rates"],
        exit_rates=input_dict["exit_rates"],
        page_values=input_dict["page_values"],
        special_day=input_dict["special_day"],
        month=input_dict["month"],
        visitor_type=input_dict["visitor_type"],
        weekend=bool(input_dict["weekend"]),
        operating_systems=input_dict["operating_systems"],
        browser=input_dict["browser"],
        region=input_dict["region"],
        traffic_type=input_dict["traffic_type"],

        device_type="web",
        ip_address="127.0.0.1",
        location="unknown",
        session_status="active",
        start_time=datetime.utcnow(),
        end_time=None
    )
    db.add(db_session)

    # Save prediction record
    db_pred = DBPrediction(
        session_id=session_id,
        will_buy=will_buy,
        buy_probability=buy_prob,
        confidence=buy_prob,
        model_version="v1",
        model_type="purchase",
        features_used=input_dict,
        shap_values=None
    )
    db.add(db_pred)

    # Save persona data
    db_persona = DBPersona(
        session_id=session_id,
        cluster_id=cluster_id,
        persona_type=persona,
        confidence=confidence,
        characteristics=persona_profiles[cluster_id]
    )
    db.add(db_persona)

    # Save incentive data
    db_incentive = DBIncentive(
        session_id=session_id,
        incentive_type=incentive,
        was_shown=True,
        was_accepted=None
    )
    db.add(db_incentive)

    db.commit()

    # ✅ Return final response
    return {
        "session_id": session_id,
        "timestamp": timestamp.isoformat(),
        "prediction": {
            "will_buy": will_buy,
            "buy_probability": round(buy_prob, 4),
            "bail_probability": round(1 - buy_prob, 4),
            "confidence": round(buy_prob * 100, 2)
        },
        "persona": {
            "cluster_id": cluster_id,
            "persona_type": persona,
            "percentage": persona_profiles[cluster_id]["percentage"]
        },
        "incentive": {
            "incentive_type": incentive,
            "confidence": round(confidence, 4)
        }
    }
