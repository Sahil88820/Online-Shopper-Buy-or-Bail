"""
Database models & engine for Smart Shopper AI

Default: SQLite for local development
Production: Set DATABASE_URL env to PostgreSQL URL

Usage:
    from database import Base, engine, SessionLocal
    Base.metadata.create_all(bind=engine)
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean,
    DateTime, JSON, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func

# ---------------------------------------------------------------------
# DB CONNECTION
# ---------------------------------------------------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./smartshopper.db"  # fallback for local testing
)

# For SQLite thread issues with FastAPI
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    name = Column(String, nullable=True)
    total_purchases = Column(Integer, default=0)
    total_spent = Column(Float, default=0.0)
    loyalty_points = Column(Integer, default=0)
    user_tier = Column(String, default="standard")
    registration_date = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)


class Session(Base):
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)

    # Full input features
    administrative = Column(Float)
    administrative_duration = Column(Float)
    informational = Column(Float)
    informational_duration = Column(Float)
    product_related = Column(Float)
    product_related_duration = Column(Float)
    bounce_rates = Column(Float)
    exit_rates = Column(Float)
    page_values = Column(Float)
    special_day = Column(Float)
    month = Column(String)
    visitor_type = Column(String)
    weekend = Column(Boolean)
    operating_systems = Column(Integer)
    browser = Column(Integer)
    region = Column(Integer)
    traffic_type = Column(Integer)

    device_type = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    location = Column(String, nullable=True)
    session_status = Column(String, default="active")
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.session_id"))
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)

    will_buy = Column(Boolean)
    buy_probability = Column(Float)
    confidence = Column(Float)
    model_version = Column(String, default="v1")
    model_type = Column(String, default="purchase")
    features_used = Column(JSON)
    shap_values = Column(JSON)
    predicted_at = Column(DateTime, default=datetime.utcnow)


class Persona(Base):
    __tablename__ = "personas"

    persona_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.session_id"))
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)

    cluster_id = Column(Integer)
    persona_type = Column(String)
    persona_color = Column(String, default="#4e79a7")
    confidence = Column(Float)
    characteristics = Column(JSON)
    assigned_at = Column(DateTime, default=datetime.utcnow)


class Incentive(Base):
    __tablename__ = "incentives"

    incentive_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.session_id"))
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)

    incentive_type = Column(String)
    category = Column(String, nullable=True)
    message = Column(String, nullable=True)
    discount_percent = Column(Float, nullable=True)
    loyalty_points = Column(Integer, nullable=True)

    was_shown = Column(Boolean, default=False)
    was_accepted = Column(Boolean, default=False)
    shown_at = Column(DateTime)
    accepted_at = Column(DateTime)


class Analytics(Base):
    __tablename__ = "analytics"

    analytics_id = Column(Integer, primary_key=True, index=True)
    snapshot_date = Column(DateTime, default=func.now())
    total_sessions = Column(Integer)
    conversion_rate = Column(Float)
    persona_distribution = Column(JSON)
    incentive_effectiveness = Column(JSON)


# ---------------------------------------------------------------------
# DB UTILS
# ---------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create DB tables if they don't exist
def init_db():
    print("🛠️ Creating database tables (if missing)...")
    Base.metadata.create_all(bind=engine)
    print("✅ DB ready!")

