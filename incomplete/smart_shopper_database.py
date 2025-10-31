"""
Smart Shopper AI - Database Models using SQLAlchemy
PostgreSQL schema for storing sessions, predictions, personas, and incentives
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, ForeignKey, Text, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os

Base = declarative_base()

# ============================================================
# Database Models
# ============================================================

class User(Base):
    """User/Customer model"""
    __tablename__ = 'users'
    
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    registration_date = Column(DateTime, default=datetime.utcnow)
    total_purchases = Column(Integer, default=0)
    total_spent = Column(Float, default=0.0)
    loyalty_points = Column(Integer, default=0)
    user_tier = Column(String(20), default='standard')  # standard, gold, platinum, vip
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sessions = relationship("Session", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")
    personas = relationship("Persona", back_populates="user")
    incentives = relationship("Incentive", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.user_id}, email={self.email}, tier={self.user_tier})>"


class Session(Base):
    """Shopping session model"""
    __tablename__ = 'sessions'
    
    session_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=True)
    session_token = Column(String(255), unique=True, nullable=False)
    
    # Session metrics (from dataset)
    administrative = Column(Integer, default=0)
    administrative_duration = Column(Float, default=0.0)
    informational = Column(Integer, default=0)
    informational_duration = Column(Float, default=0.0)
    product_related = Column(Integer, default=0)
    product_related_duration = Column(Float, default=0.0)
    bounce_rates = Column(Float, default=0.0)
    exit_rates = Column(Float, default=0.0)
    page_values = Column(Float, default=0.0)
    special_day = Column(Float, default=0.0)
    
    # Engineered features
    total_duration = Column(Float, default=0.0)
    total_pages = Column(Integer, default=0)
    engagement_score = Column(Float, default=0.0)
    product_focus_ratio = Column(Float, default=0.0)
    
    # Categorical features
    month = Column(String(20))
    visitor_type = Column(String(50))
    weekend = Column(Boolean, default=False)
    operating_systems = Column(Integer)
    browser = Column(Integer)
    region = Column(Integer)
    traffic_type = Column(Integer)
    
    # Session metadata
    device_type = Column(String(20))  # desktop, mobile, tablet
    ip_address = Column(String(50))
    location_country = Column(String(2))
    location_city = Column(String(100))
    referrer_source = Column(String(255))
    
    # Status
    session_status = Column(String(20), default='active')  # active, completed, abandoned
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    prediction = relationship("Prediction", back_populates="session", uselist=False)
    persona = relationship("Persona", back_populates="session", uselist=False)
    incentive = relationship("Incentive", back_populates="session", uselist=False)
    
    def __repr__(self):
        return f"<Session(id={self.session_id}, status={self.session_status})>"


class Prediction(Base):
    """Purchase prediction results"""
    __tablename__ = 'predictions'
    
    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('sessions.session_id'), unique=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=True)
    
    # Prediction results
    will_buy = Column(Boolean, nullable=False)
    buy_probability = Column(Float, nullable=False)
    bail_probability = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Model metadata
    model_version = Column(String(50))
    model_type = Column(String(20))  # catboost, xgboost
    features_used = Column(JSON)  # Store feature values as JSON
    
    # SHAP values (for explainability)
    shap_values = Column(JSON)  # Store top SHAP values
    top_features = Column(JSON)  # Store most influential features
    
    predicted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("Session", back_populates="prediction")
    user = relationship("User", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction(id={self.prediction_id}, will_buy={self.will_buy}, prob={self.buy_probability:.2f})>"


class Persona(Base):
    """Shopper persona/cluster assignment"""
    __tablename__ = 'personas'
    
    persona_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('sessions.session_id'), unique=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=True)
    
    # Persona details
    cluster_id = Column(Integer, nullable=False)
    persona_type = Column(String(50), nullable=False)  # Deal Hunter, Impulse Buyer, etc.
    persona_color = Column(String(20))
    confidence = Column(Float)
    
    # Cluster characteristics
    characteristics = Column(JSON)  # Key traits of this persona
    
    assigned_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("Session", back_populates="persona")
    user = relationship("User", back_populates="personas")
    
    def __repr__(self):
        return f"<Persona(id={self.persona_id}, type={self.persona_type})>"


class Incentive(Base):
    """Recommended and applied incentives"""
    __tablename__ = 'incentives'
    
    incentive_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('sessions.session_id'), unique=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=True)
    
    # Incentive details
    incentive_type = Column(String(50), nullable=False)  # discount_10, loyalty_points, etc.
    category = Column(String(20))  # discount, loyalty, shipping, urgency
    message = Column(Text)
    
    # Incentive parameters
    discount_percent = Column(Integer, default=0)
    loyalty_points = Column(Integer, default=0)
    urgency_level = Column(String(20))
    
    # Recommendation metadata
    confidence = Column(Float)
    all_probabilities = Column(JSON)  # Probabilities for all incentive types
    
    # Tracking
    was_shown = Column(Boolean, default=False)
    was_accepted = Column(Boolean, default=False)
    shown_at = Column(DateTime)
    accepted_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("Session", back_populates="incentive")
    user = relationship("User", back_populates="incentives")
    
    def __repr__(self):
        return f"<Incentive(id={self.incentive_id}, type={self.incentive_type}, accepted={self.was_accepted})>"


class Analytics(Base):
    """Aggregated analytics snapshots"""
    __tablename__ = 'analytics'
    
    analytics_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Time period
    snapshot_date = Column(DateTime, nullable=False)
    period_type = Column(String(20))  # hourly, daily, weekly
    
    # Metrics
    total_sessions = Column(Integer, default=0)
    total_predictions = Column(Integer, default=0)
    conversion_rate = Column(Float, default=0.0)
    avg_session_time = Column(Float, default=0.0)
    avg_buy_probability = Column(Float, default=0.0)
    
    # Persona distribution
    persona_distribution = Column(JSON)
    top_persona = Column(String(50))
    
    # Incentive effectiveness
    incentives_shown = Column(Integer, default=0)
    incentives_accepted = Column(Integer, default=0)
    incentive_acceptance_rate = Column(Float, default=0.0)
    incentive_distribution = Column(JSON)
    
    # Revenue (if available)
    total_revenue = Column(Float, default=0.0)
    avg_order_value = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Analytics(date={self.snapshot_date}, conversion={self.conversion_rate:.2%})>"


# ============================================================
# Database Management Functions
# ============================================================

class DatabaseManager:
    """Manage database connections and operations"""
    
    def __init__(self, database_url=None):
        if database_url is None:
            # Default to local PostgreSQL
            database_url = os.getenv(
                'DATABASE_URL',
                'postgresql://postgres:password@localhost:5432/smart_shopper_ai'
            )
        
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(self.engine)
        print("✅ Database tables created successfully!")
    
    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(self.engine)
        print("⚠️  All tables dropped!")
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    def save_prediction(self, session_data, prediction_result, persona_result, incentive_result):
        """Save a complete prediction result to database"""
        db_session = self.get_session()
        
        try:
            # Create session record
            session = Session(**session_data)
            db_session.add(session)
            db_session.flush()  # Get session_id
            
            # Create prediction record
            prediction = Prediction(
                session_id=session.session_id,
                **prediction_result
            )
            db_session.add(prediction)
            
            # Create persona record
            persona = Persona(
                session_id=session.session_id,
                **persona_result
            )
            db_session.add(persona)
            
            # Create incentive record
            incentive = Incentive(
                session_id=session.session_id,
                **incentive_result
            )
            db_session.add(incentive)
            
            db_session.commit()
            
            return {
                'session_id': session.session_id,
                'prediction_id': prediction.prediction_id,
                'persona_id': persona.persona_id,
                'incentive_id': incentive.incentive_id
            }
            
        except Exception as e:
            db_session.rollback()
            print(f"Error saving prediction: {e}")
            return None
        finally:
            db_session.close()
    
    def get_analytics_summary(self, days=7):
        """Get analytics summary for last N days"""
        from datetime import timedelta
        
        db_session = self.get_session()
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            analytics = db_session.query(Analytics).filter(
                Analytics.snapshot_date >= cutoff_date
            ).all()
            
            if not analytics:
                return None
            
            # Aggregate metrics
            summary = {
                'total_sessions': sum(a.total_sessions for a in analytics),
                'avg_conversion_rate': sum(a.conversion_rate for a in analytics) / len(analytics),
                'total_incentives_shown': sum(a.incentives_shown for a in analytics),
                'total_incentives_accepted': sum(a.incentives_accepted for a in analytics),
                'period': f'Last {days} days'
            }
            
            return summary
            
        finally:
            db_session.close()


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Initialize database
    db_manager = DatabaseManager()
    
    # Create tables
    print("Creating database tables...")
    db_manager.create_tables()
    
    # Example: Save a prediction
    session_data = {
        'session_token': 'test_session_123',
        'administrative': 5,
        'administrative_duration': 120.5,
        'product_related': 15,
        'product_related_duration': 450.3,
        'bounce_rates': 0.02,
        'exit_rates': 0.03,
        'page_values': 25.5,
        'device_type': 'desktop',
        'session_status': 'active'
    }
    
    prediction_result = {
        'will_buy': True,
        'buy_probability': 0.78,
        'bail_probability': 0.22,
        'confidence_score': 78.0,
        'model_version': 'v1.0',
        'model_type': 'catboost'
    }
    
    persona_result = {
        'cluster_id': 1,
        'persona_type': 'Impulse Buyer',
        'persona_color': 'green',
        'confidence': 0.85
    }
    
    incentive_result = {
        'incentive_type': 'loyalty_points',
        'category': 'loyalty',
        'message': 'Earn 500 bonus points!',
        'loyalty_points': 500,
        'confidence': 72.0
    }
    
    # Save to database
    result = db_manager.save_prediction(
        session_data,
        prediction_result,
        persona_result,
        incentive_result
    )
    
    if result:
        print(f"\n✅ Prediction saved successfully!")
        print(f"   Session ID: {result['session_id']}")
        print(f"   Prediction ID: {result['prediction_id']}")
        print(f"   Persona ID: {result['persona_id']}")
        print(f"   Incentive ID: {result['incentive_id']}")
