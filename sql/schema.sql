-- Cloud Kitchen Sentiment & Churn Analysis
-- Data Warehouse Schema
-- Optimized for analytics and ML feature serving

-- ============================================
-- DIMENSION TABLES
-- ============================================

CREATE TABLE dim_restaurant (
    restaurant_id INT PRIMARY KEY,
    restaurant_name VARCHAR(255) NOT NULL,
    location VARCHAR(100),
    address TEXT,
    rest_type VARCHAR(100),
    primary_cuisine VARCHAR(100),
    cuisine_count INT,
    cuisines TEXT,
    phone VARCHAR(20),
    url TEXT,
    online_order BOOLEAN,
    book_table BOOLEAN,
    listed_type VARCHAR(50),
    listed_city VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE dim_date (
    date_id INT PRIMARY KEY,
    full_date DATE NOT NULL,
    day_of_week INT,
    day_name VARCHAR(10),
    week_of_year INT,
    month INT,
    month_name VARCHAR(10),
    quarter INT,
    year INT,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN
);

CREATE TABLE dim_cuisine (
    cuisine_id INT PRIMARY KEY,
    cuisine_name VARCHAR(100) NOT NULL,
    cuisine_category VARCHAR(50),  -- e.g., 'Indian', 'Continental', 'Asian'
    is_vegetarian_friendly BOOLEAN,
    avg_preparation_time_mins INT,
    spice_level VARCHAR(20)
);

-- ============================================
-- FACT TABLES
-- ============================================

CREATE TABLE fact_restaurant_performance (
    performance_id BIGINT PRIMARY KEY,
    restaurant_id INT REFERENCES dim_restaurant(restaurant_id),
    date_id INT REFERENCES dim_date(date_id),

    -- Ratings & Votes
    overall_rating DECIMAL(2,1),
    vote_count INT,
    review_count INT,

    -- Financial
    approx_cost_two DECIMAL(10,2),
    price_category VARCHAR(20),

    -- Engagement Metrics
    avg_review_rating DECIMAL(2,1),
    review_velocity DECIMAL(5,2),  -- reviews per day
    days_since_last_review INT,

    -- Operational Flags
    has_online_order BOOLEAN,
    has_table_booking BOOLEAN,
    service_score INT,

    -- Computed Scores
    cost_efficiency_score DECIMAL(5,2),
    location_competition_index INT,

    -- Audit
    etl_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE fact_review (
    review_id BIGINT PRIMARY KEY,
    restaurant_id INT REFERENCES dim_restaurant(restaurant_id),
    date_id INT REFERENCES dim_date(date_id),

    -- Review Content
    review_text TEXT,
    review_rating DECIMAL(2,1),
    rating_header VARCHAR(20),

    -- Sentiment Scores
    vader_compound DECIMAL(5,4),
    vader_positive DECIMAL(5,4),
    vader_negative DECIMAL(5,4),
    vader_neutral DECIMAL(5,4),
    textblob_polarity DECIMAL(5,4),
    textblob_subjectivity DECIMAL(5,4),
    sentiment_label VARCHAR(10),  -- 'Positive', 'Negative', 'Neutral'

    -- Aspect Analysis
    aspects TEXT,  -- JSON array of aspects mentioned
    aspect_count INT,

    -- Operational Issue Flags
    has_delivery_issue BOOLEAN DEFAULT FALSE,
    has_food_quality_issue BOOLEAN DEFAULT FALSE,
    has_packaging_issue BOOLEAN DEFAULT FALSE,
    has_service_issue BOOLEAN DEFAULT FALSE,
    has_hygiene_issue BOOLEAN DEFAULT FALSE,
    has_wrong_order_issue BOOLEAN DEFAULT FALSE,
    any_operational_issue BOOLEAN DEFAULT FALSE,

    -- Agreement Analysis
    sentiment_rating_agreement VARCHAR(10),  -- 'Agree', 'Disagree'

    -- Audit
    etl_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- AGGREGATE TABLES (Materialized Views)
-- ============================================

CREATE TABLE agg_restaurant_sentiment_daily (
    restaurant_id INT REFERENCES dim_restaurant(restaurant_id),
    date_id INT REFERENCES dim_date(date_id),
    avg_sentiment DECIMAL(5,4),
    sentiment_std DECIMAL(5,4),
    positive_review_count INT,
    negative_review_count INT,
    neutral_review_count INT,
    total_reviews INT,
    negative_review_pct DECIMAL(5,2),
    operational_issue_count INT,
    operational_issue_rate DECIMAL(5,2),
    PRIMARY KEY (restaurant_id, date_id)
);

CREATE TABLE agg_restaurant_churn_risk (
    restaurant_id INT PRIMARY KEY REFERENCES dim_restaurant(restaurant_id),
    last_calculated_date DATE,

    -- Risk Features
    churn_risk_score INT,  -- 0-5 composite score
    churn_probability DECIMAL(5,4),
    risk_category VARCHAR(20),  -- 'Low', 'Medium', 'High', 'Critical'

    -- Key Indicators
    rating_trend VARCHAR(10),  -- 'Improving', 'Stable', 'Declining'
    sentiment_trend VARCHAR(10),
    review_velocity_trend VARCHAR(10),
    operational_issue_spike BOOLEAN,

    -- Intervention Tracking
    last_intervention_date DATE,
    intervention_type VARCHAR(50),
    intervention_outcome VARCHAR(20),

    -- Audit
    model_version VARCHAR(20),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- INDEXES FOR PERFORMANCE
-- ============================================

CREATE INDEX idx_review_restaurant ON fact_review(restaurant_id);
CREATE INDEX idx_review_date ON fact_review(date_id);
CREATE INDEX idx_review_sentiment ON fact_review(sentiment_label);
CREATE INDEX idx_review_operational ON fact_review(any_operational_issue) WHERE any_operational_issue = TRUE;
CREATE INDEX idx_performance_restaurant_date ON fact_restaurant_performance(restaurant_id, date_id);
CREATE INDEX idx_restaurant_location ON dim_restaurant(location);
CREATE INDEX idx_restaurant_type ON dim_restaurant(rest_type);

-- ============================================
-- ANALYTICS VIEWS
-- ============================================

CREATE VIEW vw_restaurant_health_score AS
SELECT 
    r.restaurant_id,
    r.restaurant_name,
    r.location,
    p.overall_rating,
    p.review_count,
    p.days_since_last_review,
    s.avg_sentiment,
    s.negative_review_pct,
    s.operational_issue_rate,
    c.churn_probability,
    c.risk_category,
    -- Composite health score (0-100)
    (
        (p.overall_rating * 10) + 
        (LEAST(p.review_count, 100)) + 
        (CASE WHEN p.days_since_last_review < 30 THEN 20 ELSE 0 END) +
        ((s.avg_sentiment + 1) * 25) -
        (s.negative_review_pct * 0.5) -
        (s.operational_issue_rate * 30)
    )::INT AS health_score
FROM dim_restaurant r
LEFT JOIN fact_restaurant_performance p ON r.restaurant_id = p.restaurant_id
LEFT JOIN agg_restaurant_sentiment_daily s ON r.restaurant_id = s.restaurant_id
LEFT JOIN agg_restaurant_churn_risk c ON r.restaurant_id = c.restaurant_id;

CREATE VIEW vw_location_performance_summary AS
SELECT 
    r.location,
    COUNT(DISTINCT r.restaurant_id) AS restaurant_count,
    AVG(p.overall_rating) AS avg_rating,
    AVG(p.approx_cost_two) AS avg_cost,
    SUM(p.vote_count) AS total_votes,
    AVG(s.avg_sentiment) AS avg_sentiment,
    AVG(c.churn_probability) AS avg_churn_risk,
    COUNT(DISTINCT CASE WHEN c.risk_category IN ('High', 'Critical') THEN r.restaurant_id END) AS at_risk_count
FROM dim_restaurant r
LEFT JOIN fact_restaurant_performance p ON r.restaurant_id = p.restaurant_id
LEFT JOIN agg_restaurant_sentiment_daily s ON r.restaurant_id = s.restaurant_id
LEFT JOIN agg_restaurant_churn_risk c ON r.restaurant_id = c.restaurant_id
GROUP BY r.location;

CREATE VIEW vw_operational_issue_trends AS
SELECT 
    d.month_name,
    d.year,
    COUNT(CASE WHEN rv.has_delivery_issue THEN 1 END) AS delivery_issues,
    COUNT(CASE WHEN rv.has_food_quality_issue THEN 1 END) AS food_quality_issues,
    COUNT(CASE WHEN rv.has_packaging_issue THEN 1 END) AS packaging_issues,
    COUNT(CASE WHEN rv.has_service_issue THEN 1 END) AS service_issues,
    COUNT(CASE WHEN rv.has_hygiene_issue THEN 1 END) AS hygiene_issues,
    COUNT(*) AS total_reviews,
    ROUND(
        COUNT(CASE WHEN rv.any_operational_issue THEN 1 END) * 100.0 / COUNT(*), 2
    ) AS operational_issue_pct
FROM fact_review rv
JOIN dim_date d ON rv.date_id = d.date_id
GROUP BY d.month_name, d.year
ORDER BY d.year, d.month;

-- ============================================
-- STORED PROCEDURES
-- ============================================

CREATE OR REPLACE FUNCTION update_churn_risk_batch()
RETURNS void AS $$
BEGIN
    -- Update churn risk scores based on latest data
    UPDATE agg_restaurant_churn_risk c
    SET 
        churn_risk_score = (
            SELECT 
                CASE 
                    WHEN p.overall_rating < 3.0 AND s.negative_review_pct > 40 THEN 2 ELSE 0 END +
                CASE WHEN s.operational_issue_rate > 0.30 THEN 2 ELSE 0 END +
                CASE WHEN p.days_since_last_review > 60 THEN 1 ELSE 0 END +
                CASE WHEN s.avg_sentiment < -0.2 THEN 1 ELSE 0 END
            FROM fact_restaurant_performance p
            JOIN agg_restaurant_sentiment_daily s ON p.restaurant_id = s.restaurant_id
            WHERE p.restaurant_id = c.restaurant_id
        ),
        last_calculated_date = CURRENT_DATE,
        calculated_at = CURRENT_TIMESTAMP
    WHERE EXISTS (
        SELECT 1 FROM fact_restaurant_performance p 
        WHERE p.restaurant_id = c.restaurant_id
    );
END;
$$ LANGUAGE plpgsql;
