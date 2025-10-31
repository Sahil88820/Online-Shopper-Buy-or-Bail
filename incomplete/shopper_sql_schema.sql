-- Shopper Behavior Predictor Database Schema
-- Database: PostgreSQL 14+ (compatible with MySQL with minor modifications)

-- ============================================================
-- TABLE: users
-- Stores user/customer information
-- ============================================================
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_purchases INTEGER DEFAULT 0,
    total_spent DECIMAL(10, 2) DEFAULT 0.00,
    loyalty_points INTEGER DEFAULT 0,
    user_tier VARCHAR(20) DEFAULT 'standard', -- standard, gold, platinum, vip
    last_active TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_tier ON users(user_tier);

-- ============================================================
-- TABLE: sessions
-- Stores browsing session data for predictions
-- ============================================================
CREATE TABLE sessions (
    session_id BIGSERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    time_on_site INTEGER, -- seconds
    pages_visited INTEGER DEFAULT 0,
    device_type VARCHAR(20), -- desktop, mobile, tablet
    browser VARCHAR(50),
    ip_address INET,
    location_country VARCHAR(2),
    location_city VARCHAR(100),
    referrer_source VARCHAR(255),
    session_status VARCHAR(20) DEFAULT 'active', -- active, completed, abandoned
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_start_time ON sessions(start_time);
CREATE INDEX idx_sessions_status ON sessions(session_status);
CREATE INDEX idx_sessions_device ON sessions(device_type);

-- ============================================================
-- TABLE: carts
-- Stores shopping cart information
-- ============================================================
CREATE TABLE carts (
    cart_id SERIAL PRIMARY KEY,
    session_id BIGINT REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    cart_value DECIMAL(10, 2) DEFAULT 0.00,
    item_count INTEGER DEFAULT 0,
    cart_status VARCHAR(20) DEFAULT 'active', -- active, abandoned, converted
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    abandoned_at TIMESTAMP,
    converted_at TIMESTAMP
);

CREATE INDEX idx_carts_session ON carts(session_id);
CREATE INDEX idx_carts_user ON carts(user_id);
CREATE INDEX idx_carts_status ON carts(cart_status);

-- ============================================================
-- TABLE: cart_items
-- Stores individual items in shopping carts
-- ============================================================
CREATE TABLE cart_items (
    cart_item_id SERIAL PRIMARY KEY,
    cart_id INTEGER REFERENCES carts(cart_id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL,
    product_name VARCHAR(255),
    quantity INTEGER DEFAULT 1,
    unit_price DECIMAL(10, 2),
    total_price DECIMAL(10, 2),
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cart_items_cart ON cart_items(cart_id);
CREATE INDEX idx_cart_items_product ON cart_items(product_id);

-- ============================================================
-- TABLE: predictions
-- Stores ML model predictions for each session
-- ============================================================
CREATE TABLE predictions (
    prediction_id SERIAL PRIMARY KEY,
    session_id BIGINT REFERENCES sessions(session_id) ON DELETE CASCADE,
    will_buy BOOLEAN,
    confidence_score INTEGER, -- 0-100
    probability DECIMAL(5, 4), -- 0.0000-1.0000
    model_version VARCHAR(50),
    features_json JSONB, -- Store all feature values as JSON
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_session ON predictions(session_id);
CREATE INDEX idx_predictions_will_buy ON predictions(will_buy);
CREATE INDEX idx_predictions_confidence ON predictions(confidence_score);
CREATE INDEX idx_predictions_predicted_at ON predictions(predicted_at);

-- ============================================================
-- TABLE: personas
-- Stores persona cluster assignments
-- ============================================================
CREATE TABLE personas (
    persona_id SERIAL PRIMARY KEY,
    session_id BIGINT REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
    persona_type VARCHAR(50), -- VIP Loyalist, Research Shopper, Quick Buyer, Browser, Casual Visitor
    persona_color VARCHAR(20),
    cluster_id INTEGER,
    confidence DECIMAL(5, 4),
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_personas_session ON personas(session_id);
CREATE INDEX idx_personas_user ON personas(user_id);
CREATE INDEX idx_personas_type ON personas(persona_type);

-- ============================================================
-- TABLE: incentives
-- Stores recommended and applied incentives
-- ============================================================
CREATE TABLE incentives (
    incentive_id SERIAL PRIMARY KEY,
    session_id BIGINT REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
    incentive_type VARCHAR(20), -- discount, loyalty, urgency, none
    message TEXT,
    discount_percentage INTEGER DEFAULT 0,
    loyalty_points INTEGER DEFAULT 0,
    priority VARCHAR(10), -- low, medium, high
    was_shown BOOLEAN DEFAULT FALSE,
    was_accepted BOOLEAN DEFAULT FALSE,
    shown_at TIMESTAMP,
    accepted_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_incentives_session ON incentives(session_id);
CREATE INDEX idx_incentives_user ON incentives(user_id);
CREATE INDEX idx_incentives_type ON incentives(incentive_type);
CREATE INDEX idx_incentives_accepted ON incentives(was_accepted);

-- ============================================================
-- TABLE: conversions
-- Stores successful purchase conversions
-- ============================================================
CREATE TABLE conversions (
    conversion_id SERIAL PRIMARY KEY,
    session_id BIGINT REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    cart_id INTEGER REFERENCES carts(cart_id) ON DELETE CASCADE,
    order_id VARCHAR(100) UNIQUE,
    total_amount DECIMAL(10, 2),
    discount_applied DECIMAL(10, 2) DEFAULT 0.00,
    incentive_id INTEGER REFERENCES incentives(incentive_id) ON DELETE SET NULL,
    payment_method VARCHAR(50),
    converted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_conversions_session ON conversions(session_id);
CREATE INDEX idx_conversions_user ON conversions(user_id);
CREATE INDEX idx_conversions_order ON conversions(order_id);
CREATE INDEX idx_conversions_converted_at ON conversions(converted_at);

-- ============================================================
-- TABLE: analytics_snapshots
-- Stores periodic analytics snapshots for dashboard
-- ============================================================
CREATE TABLE analytics_snapshots (
    snapshot_id SERIAL PRIMARY KEY,
    total_sessions INTEGER,
    conversion_rate DECIMAL(5, 2),
    avg_session_time INTEGER,
    top_persona VARCHAR(50),
    total_revenue DECIMAL(12, 2),
    incentives_shown INTEGER,
    incentives_accepted INTEGER,
    snapshot_date DATE,
    snapshot_hour INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analytics_date ON analytics_snapshots(snapshot_date);
CREATE INDEX idx_analytics_hour ON analytics_snapshots(snapshot_hour);

-- ============================================================
-- TABLE: model_performance
-- Tracks ML model performance metrics
-- ============================================================
CREATE TABLE model_performance (
    performance_id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    accuracy DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    total_predictions INTEGER,
    correct_predictions INTEGER,
    evaluation_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_version ON model_performance(model_version);
CREATE INDEX idx_model_eval_date ON model_performance(evaluation_date);

-- ============================================================
-- VIEWS
-- ============================================================

-- Real-time dashboard view
CREATE VIEW dashboard_realtime AS
SELECT 
    COUNT(DISTINCT s.session_id) as total_sessions,
    ROUND(
        (COUNT(DISTINCT c.conversion_id)::NUMERIC / 
        NULLIF(COUNT(DISTINCT s.session_id), 0) * 100), 2
    ) as conversion_rate,
    ROUND(AVG(s.time_on_site)) as avg_session_time,
    (
        SELECT persona_type 
        FROM personas 
        GROUP BY persona_type 
        ORDER BY COUNT(*) DESC 
        LIMIT 1
    ) as top_persona,
    COALESCE(SUM(c.total_amount), 0) as total_revenue
FROM sessions s
LEFT JOIN conversions c ON s.session_id = c.session_id
WHERE s.start_time >= NOW() - INTERVAL '24 hours';

-- Incentive performance view
CREATE VIEW incentive_performance AS
SELECT 
    i.incentive_type,
    COUNT(*) as times_shown,
    COUNT(CASE WHEN i.was_accepted THEN 1 END) as times_accepted,
    ROUND(
        COUNT(CASE WHEN i.was_accepted THEN 1 END)::NUMERIC / 
        NULLIF(COUNT(*), 0) * 100, 2
    ) as acceptance_rate,
    COALESCE(AVG(c.total_amount), 0) as avg_order_value
FROM incentives i
LEFT JOIN conversions c ON i.incentive_id = c.incentive_id
WHERE i.was_shown = TRUE
GROUP BY i.incentive_type;

-- User persona distribution
CREATE VIEW persona_distribution AS
SELECT 
    persona_type,
    COUNT(*) as count,
    ROUND(
        COUNT(*)::NUMERIC / 
        (SELECT COUNT(*) FROM personas) * 100, 2
    ) as percentage,
    COALESCE(AVG(c.total_amount), 0) as avg_lifetime_value
FROM personas p
LEFT JOIN users u ON p.user_id = u.user_id
LEFT JOIN conversions c ON u.user_id = c.user_id
GROUP BY persona_type
ORDER BY count DESC;

-- ============================================================
-- FUNCTIONS
-- ============================================================

-- Calculate session metrics
CREATE OR REPLACE FUNCTION calculate_session_metrics(p_session_id BIGINT)
RETURNS TABLE (
    time_on_site INTEGER,
    pages_visited INTEGER,
    cart_value DECIMAL,
    previous_purchases INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.time_on_site,
        s.pages_visited,
        COALESCE(ca.cart_value, 0) as cart_value,
        COALESCE(u.total_purchases, 0) as previous_purchases
    FROM sessions s
    LEFT JOIN carts ca ON s.session_id = ca.session_id
    LEFT JOIN users u ON s.user_id = u.user_id
    WHERE s.session_id = p_session_id;
END;
$$ LANGUAGE plpgsql;

-- Update analytics snapshot (call hourly via cron)
CREATE OR REPLACE FUNCTION update_analytics_snapshot()
RETURNS VOID AS $$
BEGIN
    INSERT INTO analytics_snapshots (
        total_sessions,
        conversion_rate,
        avg_session_time,
        top_persona,
        total_revenue,
        incentives_shown,
        incentives_accepted,
        snapshot_date,
        snapshot_hour
    )
    SELECT 
        COUNT(DISTINCT s.session_id),
        ROUND(
            (COUNT(DISTINCT c.conversion_id)::NUMERIC / 
            NULLIF(COUNT(DISTINCT s.session_id), 0) * 100), 2
        ),
        ROUND(AVG(s.time_on_site)),
        (SELECT persona_type FROM personas GROUP BY persona_type ORDER BY COUNT(*) DESC LIMIT 1),
        COALESCE(SUM(c.total_amount), 0),
        (SELECT COUNT(*) FROM incentives WHERE was_shown = TRUE),
        (SELECT COUNT(*) FROM incentives WHERE was_accepted = TRUE),
        CURRENT_DATE,
        EXTRACT(HOUR FROM CURRENT_TIMESTAMP)
    FROM sessions s
    LEFT JOIN conversions c ON s.session_id = c.session_id
    WHERE s.start_time >= NOW() - INTERVAL '1 hour';
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- SAMPLE DATA (for testing)
-- ============================================================

-- Insert sample users
INSERT INTO users (email, first_name, last_name, total_purchases, user_tier) VALUES
('john.doe@email.com', 'John', 'Doe', 8, 'platinum'),
('jane.smith@email.com', 'Jane', 'Smith', 3, 'gold'),
('bob.wilson@email.com', 'Bob', 'Wilson', 0, 'standard');

-- Sample queries for analytics
COMMENT ON VIEW dashboard_realtime IS 'Real-time dashboard metrics for the last 24 hours';
COMMENT ON VIEW incentive_performance IS 'Performance metrics for different incentive types';
COMMENT ON VIEW persona_distribution IS 'Distribution of user personas and their lifetime value';