import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text

def get_db_engine():
    # Priority order: 1. env variable, 2. hardcoded connection string fallback
    db_url = os.environ.get(
        "DATABASE_URL", 
        "postgresql://neondb_owner:npg_tSRYv5mXKs4P@ep-misty-rice-ap6vsh1m-pooler.c-7.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    )
    # Fix render's postgres:// to postgresql:// format if needed
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    return create_engine(db_url, pool_pre_ping=True)

def ingest_data():
    engine = get_db_engine()
    
    print("Testing connection to cloud Neon PostgreSQL...")
    with engine.connect() as conn:
        res = conn.execute(text("SELECT version();")).fetchone()
        print(f"Connected to Neon! Version: {res[0]}")
    
    # 1. Ingest Restaurants
    restaurants_csv = 'data/restaurants_with_churn_features.csv'
    if os.path.exists(restaurants_csv):
        print(f"Reading {restaurants_csv}...")
        df_rest = pd.read_csv(restaurants_csv)
        print(f"Writing {len(df_rest)} restaurants to Neon...")
        # Upload using replacement
        df_rest.to_sql('restaurants', engine, index=False, if_exists='replace', method='multi', chunksize=5000)
        
        # Build indexes on Postgres
        print("Building indexes on 'restaurants' table...")
        with engine.connect() as conn:
            # Drop old indexes if they exist
            conn.execute(text("DROP INDEX IF EXISTS idx_rest_location;"))
            conn.execute(text("DROP INDEX IF EXISTS idx_rest_cuisine;"))
            conn.execute(text("DROP INDEX IF EXISTS idx_rest_id;"))
            # Create new ones
            conn.execute(text("CREATE INDEX idx_rest_location ON restaurants(location);"))
            conn.execute(text("CREATE INDEX idx_rest_cuisine ON restaurants(primary_cuisine);"))
            conn.execute(text("CREATE INDEX idx_rest_id ON restaurants(restaurant_id);"))
            conn.commit()
        print("Restaurant ingestion and indexing completed successfully.")
    else:
        print(f"Error: {restaurants_csv} not found!")

    # 2. Ingest Reviews
    reviews_csv = 'data/reviews_with_sentiment.csv'
    if os.path.exists(reviews_csv):
        print(f"Reading {reviews_csv}...")
        reviews_cols = ['restaurant_id', 'restaurant_name', 'review_text', 'review_rating', 'sentiment_label',
                        'delivery_delay', 'food_quality', 'packaging', 'service', 'hygiene', 'wrong_order']
        
        # Load in chunks to save memory during ingestion
        df_rev = pd.read_csv(reviews_csv, usecols=lambda c: c in reviews_cols)
        
        # Ensure booleans are converted to SQL Booleans or Integers (0/1)
        bool_cols = ['delivery_delay', 'food_quality', 'packaging', 'service', 'hygiene', 'wrong_order']
        for col in bool_cols:
            if col in df_rev.columns:
                df_rev[col] = df_rev[col].fillna(0).astype(int)

        print(f"Writing {len(df_rev)} reviews to Neon (in chunks)...")
        # Write to PostgreSQL
        df_rev.to_sql('reviews', engine, index=False, if_exists='replace', method='multi', chunksize=5000)
        
        # Build indexes on Postgres
        print("Building indexes on 'reviews' table...")
        with engine.connect() as conn:
            conn.execute(text("DROP INDEX IF EXISTS idx_rev_rest_id;"))
            conn.execute(text("DROP INDEX IF EXISTS idx_rev_sentiment;"))
            conn.execute(text("CREATE INDEX idx_rev_rest_id ON reviews(restaurant_id);"))
            conn.execute(text("CREATE INDEX idx_rev_sentiment ON reviews(sentiment_label);"))
            conn.commit()
        print("Review ingestion and indexing completed successfully.")
    else:
        print(f"Error: {reviews_csv} not found!")

    print("Neon PostgreSQL Database ingestion completed successfully!")

if __name__ == '__main__':
    ingest_data()
