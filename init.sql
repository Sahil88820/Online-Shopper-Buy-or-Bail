-- Create database if it doesn't exist
CREATE DATABASE smart_shopper;

-- Connect to database
\c smart_shopper

-- Create tables
CREATE TABLE IF NOT EXISTS shoppers (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR UNIQUE NOT NULL,
    administrative_pages INTEGER NOT NULL,
    informational_pages INTEGER NOT NULL,
    product_pages INTEGER NOT NULL,
    bounce_rates FLOAT NOT NULL,
    exit_rates FLOAT NOT NULL,
    page_values FLOAT NOT NULL,
    special_day FLOAT NOT NULL,
    month VARCHAR NOT NULL,
    weekend BOOLEAN NOT NULL,
    operating_system VARCHAR NOT NULL,
    browser VARCHAR NOT NULL,
    region VARCHAR NOT NULL,
    traffic_type INTEGER NOT NULL,
    visitor_type VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    shopper_id INTEGER REFERENCES shoppers(id),
    purchase_probability FLOAT NOT NULL,
    predicted_persona INTEGER NOT NULL,
    recommended_incentive VARCHAR NOT NULL,
    actual_purchase BOOLEAN,
    incentive_accepted BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_shoppers_session_id ON shoppers(session_id);
CREATE INDEX idx_predictions_shopper_id ON predictions(shopper_id);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);