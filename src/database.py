"""
Database models and SQLAlchemy setup.
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "smart_shopper")

# Create database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for declarative models
Base = declarative_base()

class Shopper(Base):
    __tablename__ = "shoppers"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    administrative_pages = Column(Integer)
    informational_pages = Column(Integer)
    product_pages = Column(Integer)
    bounce_rates = Column(Float)
    exit_rates = Column(Float)
    page_values = Column(Float)
    special_day = Column(Float)
    month = Column(String)
    weekend = Column(Boolean)
    operating_system = Column(String)
    browser = Column(String)
    region = Column(String)
    traffic_type = Column(Integer)
    visitor_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="shopper")
    
class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    shopper_id = Column(Integer, ForeignKey("shoppers.id"))
    purchase_probability = Column(Float)
    predicted_persona = Column(Integer)
    recommended_incentive = Column(String)
    actual_purchase = Column(Boolean, nullable=True)
    incentive_accepted = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    shopper = relationship("Shopper", back_populates="predictions")

def get_db():
    """Database session context manager."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    # Initialize database
    init_db()