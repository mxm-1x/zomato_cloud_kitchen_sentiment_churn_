import os
# Load local .env file if it exists
if os.path.exists('.env'):
    with open('.env') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ[k] = v.strip('"\'')

import sys
import io

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

def copy_df_to_postgres(df, table_name, engine):
    """Highly optimized COPY streaming for extremely fast WAN uploads."""
    print(f"Initializing COPY streaming for table '{table_name}'...")
    
    # 1. Create table structure using Pandas (without inserting any rows)
    df.head(0).to_sql(table_name, engine, if_exists='replace', index=False)
    
    # 2. Establish raw connection to execute COPY
    connection = engine.raw_connection()
    try:
        cursor = connection.cursor()
        
        # 3. Write DataFrame to memory buffer as tab-separated values
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
        output.seek(0)
        
        # 4. Execute copy query
        copy_query = f"COPY {table_name} FROM STDIN WITH CSV DELIMITER '\t' NULL '\\N'"
        cursor.copy_expert(copy_query, output)
        connection.commit()
        print(f"Successfully streamed {len(df)} rows to '{table_name}'.")
    except Exception as e:
        connection.rollback()
        print(f"Error during COPY streaming: {e}")
        raise e
    finally:
        connection.close()

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
        
        # Stream restaurants
        copy_df_to_postgres(df_rest, 'restaurants', engine)
        
        # Build indexes on Postgres
        print("Building indexes on 'restaurants' table...")
        with engine.connect() as conn:
            conn.execute(text("DROP INDEX IF EXISTS idx_rest_location;"))
            conn.execute(text("DROP INDEX IF EXISTS idx_rest_cuisine;"))
            conn.execute(text("DROP INDEX IF EXISTS idx_rest_id;"))
            conn.execute(text("CREATE INDEX idx_rest_location ON restaurants(location);"))
            conn.execute(text("CREATE INDEX idx_rest_cuisine ON restaurants(primary_cuisine);"))
            conn.execute(text("CREATE INDEX idx_rest_id ON restaurants(restaurant_id);"))
            conn.commit()
        print("Restaurant indexes built successfully.")
    else:
        print(f"Error: {restaurants_csv} not found!")

    # 2. Ingest Reviews
    reviews_csv = 'data/reviews_with_sentiment.csv'
    if os.path.exists(reviews_csv):
        print(f"Reading {reviews_csv}...")
        reviews_cols = ['restaurant_id', 'restaurant_name', 'review_text', 'review_rating', 'sentiment_label',
                        'delivery_delay', 'food_quality', 'packaging', 'service', 'hygiene', 'wrong_order']
        
        # Load required columns
        df_rev = pd.read_csv(reviews_csv, usecols=lambda c: c in reviews_cols)
        
        # Ensure booleans are converted to Integers (0/1) for SQL compatibility
        bool_cols = ['delivery_delay', 'food_quality', 'packaging', 'service', 'hygiene', 'wrong_order']
        for col in bool_cols:
            if col in df_rev.columns:
                df_rev[col] = df_rev[col].fillna(0).astype(int)

        # Stream reviews
        copy_df_to_postgres(df_rev, 'reviews', engine)
        
        # Build indexes on Postgres
        print("Building indexes on 'reviews' table...")
        with engine.connect() as conn:
            conn.execute(text("DROP INDEX IF EXISTS idx_rev_rest_id;"))
            conn.execute(text("DROP INDEX IF EXISTS idx_rev_sentiment;"))
            conn.execute(text("CREATE INDEX idx_rev_rest_id ON reviews(restaurant_id);"))
            conn.execute(text("CREATE INDEX idx_rev_sentiment ON reviews(sentiment_label);"))
            conn.commit()
        print("Review indexes built successfully.")
    else:
        print(f"Error: {reviews_csv} not found!")

    print("Neon PostgreSQL Database ingestion completed successfully!")

if __name__ == '__main__':
    ingest_data()
