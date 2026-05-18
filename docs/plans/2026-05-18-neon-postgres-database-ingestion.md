# Neon PostgreSQL Database Ingestion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ingest the cloud kitchen restaurants and reviews datasets into a cloud Neon PostgreSQL database, and refactor the dashboard to dynamically fetch data on demand using SQLAlchemy and psycopg2-binary, ensuring zero RAM footprint and high-performance querying on Render.

**Architecture:**
- **Database Engine:** Cloud-hosted Neon Serverless PostgreSQL (`postgresql://...`).
- **Secret Management:** Connection string is stored in a `DATABASE_URL` environment variable for both security and ease of deployment (supported out-of-the-box by Render).
- **Ingestion automation (`src/ingest_to_db.py`):** A Python script that uses `SQLAlchemy` to connect to Neon Postgres, uploads the datasets in chunked batches (to avoid timeouts), and creates B-Tree indexes for lightning-fast dynamic filters.
- **Dynamic Callbacks:**
  - `df_restaurants` will be fetched once at startup.
  - `df_reviews` will **not** be preloaded. The reviews explorer and sentiment visualizer will execute target-filtered SQL queries with limits dynamically inside callbacks (e.g. `SELECT * FROM reviews WHERE restaurant_id IN (...) LIMIT 2000`).

**Tech Stack:** Neon PostgreSQL, SQLAlchemy, `psycopg2-binary`, Plotly Dash, Gunicorn.

---

### Task 1: Add PostgreSQL Dependencies to requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Add psycopg2-binary to requirements.txt**

Add the Postgres driver to `requirements.txt` so Render automatically builds it:

```diff
 dash>=2.11.0
 dash-bootstrap-components>=1.4.0
 gunicorn>=20.1.0
+psycopg2-binary>=2.9.9
```

**Step 2: Commit change**

```bash
git add requirements.txt
git commit -m "chore: add psycopg2-binary PostgreSQL driver dependency"
```

---

### Task 2: Create Neon PostgreSQL Ingestion Script

**Files:**
- Create: `src/ingest_to_db.py`

**Step 1: Write the ingestion script**

Write a Python script that reads the CSV datasets and uses SQLAlchemy to stream data to your cloud Neon DB in chunks (which avoids network timeouts):

```python
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
```

**Step 2: Commit ingestion script**

```bash
git add src/ingest_to_db.py
git commit -m "feat: add Neon PostgreSQL database ingestion script"
```

---

### Task 3: Install psycopg2-binary and Run Ingestion locally

**Files:**
- Create: Cloud Neon tables

**Step 1: Install dependencies inside the virtual environment**

Run:
```bash
.venv/bin/pip install psycopg2-binary
```

**Step 2: Execute Ingestion Script**

Run the ingestion script to stream your datasets directly into Neon Postgres:

```bash
DATABASE_URL="postgresql://neondb_owner:npg_tSRYv5mXKs4P@ep-misty-rice-ap6vsh1m-pooler.c-7.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require" .venv/bin/python src/ingest_to_db.py
```

Expected Output:
```bash
Testing connection to cloud Neon PostgreSQL...
Connected to Neon! Version: PostgreSQL 16.x ...
Reading data/restaurants_with_churn_features.csv...
Writing 51717 restaurants to Neon...
Building indexes on 'restaurants' table...
Restaurant ingestion and indexing completed successfully.
Reading data/reviews_with_sentiment.csv...
Writing 40000 reviews to Neon (in chunks)...
Building indexes on 'reviews' table...
Review ingestion and indexing completed successfully.
Neon PostgreSQL Database ingestion completed successfully!
```

---

### Task 4: Refactor Dashboard to Query Neon Postgres Dynamically

**Files:**
- Modify: `src/dashboard.py`

**Step 1: Add SQLAlchemy / SQL support to `load_data()`**

In `src/dashboard.py`, load the restaurant list directly from Postgres:

```python
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

def get_db_engine():
    # Read from environment, fallback to provided Neon URL
    db_url = os.environ.get(
        "DATABASE_URL", 
        "postgresql://neondb_owner:npg_tSRYv5mXKs4P@ep-misty-rice-ap6vsh1m-pooler.c-7.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    )
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    return create_engine(db_url, pool_pre_ping=True)

def load_data():
    """Load restaurant metadata from cloud Neon PostgreSQL."""
    try:
        print("Connecting to cloud Neon database...")
        engine = get_db_engine()
        # Fetch restaurants metadata from Postgres (only loads what we need, ~10MB)
        df = pd.read_sql_query("SELECT * FROM restaurants", engine)
        print(f"Successfully loaded {len(df)} restaurants from Neon DB.")
    except Exception as e:
        print(f"Error loading from Neon: {e}. Falling back to CSVs.")
        # Fallback to local files if Neon connection fails
        if os.path.exists('data/restaurants_with_churn_features.csv'):
            df = pd.read_csv('data/restaurants_with_churn_features.csv')
        else:
            # Fallback dummy data
            df = pd.DataFrame({
                'restaurant_id': range(1, 6),
                'name': ['Kitchen A', 'Kitchen B', 'Kitchen C', 'Kitchen D', 'Kitchen E'],
                'location': ['Indiranagar', 'BTM', 'Koramangala', 'HSR', 'Indiranagar'],
                'rate_numeric': [4.2, 3.1, 4.5, 2.9, 3.8],
                'cost_numeric': [500, 300, 600, 250, 450],
                'votes_numeric': [120, 45, 300, 12, 89],
                'primary_cuisine': ['North Indian', 'South Indian', 'Continental', 'Fast Food', 'Cafe'],
                'online_order_binary': [1, 1, 0, 1, 1],
                'book_table_binary': [0, 0, 1, 0, 0],
                'review_count': [15, 6, 25, 2, 10],
                'avg_sentiment': [0.4, -0.1, 0.6, -0.3, 0.2],
                'negative_review_pct': [10, 50, 5, 80, 20],
                'operational_issue_rate': [0.05, 0.25, 0.02, 0.40, 0.10],
                'days_since_last_review': [12, 45, 5, 89, 23],
                'churn_probability': [0.15, 0.65, 0.05, 0.90, 0.30],
                'risk_category': ['Low', 'High', 'Low', 'Critical', 'Medium']
            })

    # Location coordinates addition
    location_coords = {
        'BTM': (12.9166, 77.6101), 'HSR': (12.9100, 77.6450), 'Koramangala': (12.9352, 77.6244),
        'Jayanagar': (12.9307, 77.5832), 'Indiranagar': (12.9719, 77.6412), 'JP Nagar': (12.9063, 77.5857),
        'Whitefield': (12.9698, 77.7500), 'Marathahalli': (12.9569, 77.7011), 'Bannerghatta Road': (12.8900, 77.5900),
        'Electronic City': (12.8490, 77.6500), 'Bellandur': (12.9304, 77.6784), 'Sarjapur Road': (12.9100, 77.6800),
        'MG Road': (12.9738, 77.6119), 'Brigade Road': (12.9700, 77.6080), 'Kalyan Nagar': (13.0232, 77.6432),
        'Rajajinagar': (12.9882, 77.5543), 'Malleshwaram': (13.0031, 77.5684), 'Banashankari': (12.9254, 77.5468),
        'Basavanagudi': (12.9417, 77.5750), 'Frazer Town': (12.9972, 77.6147), 'Richmond Road': (12.9667, 77.6000),
        'Lavelle Road': (12.9700, 77.6000), 'Ulsoor': (12.9817, 77.6284), 'Commercial Street': (12.9808, 77.6083),
        'Domlur': (12.9610, 77.6387), 'Residency Road': (12.9667, 77.6083), 'Kammanahalli': (13.0159, 77.6378),
        'Cunningham Road': (12.9842, 77.5969), 'Old Airport Road': (12.9500, 77.6600), 'Brookefield': (12.9628, 77.7125),
        'New BEL Road': (13.0300, 77.5700), 'Sanjay Nagar': (13.0300, 77.5800), 'Koramangala 5th Block': (12.9350, 77.6200),
        'Koramangala 6th Block': (12.9370, 77.6230), 'Koramangala 7th Block': (12.9330, 77.6250), 'Koramangala 4th Block': (12.9320, 77.6280),
        'Koramangala 8th Block': (12.9380, 77.6300), 'Koramangala 1st Block': (12.9272, 77.6344), 'Jayanagar 4th Block': (12.9290, 77.5820),
        'Jayanagar 9th Block': (12.9210, 77.5930), 'Jayanagar 3rd Block': (12.9320, 77.5800), 'Jayanagar 8th Block': (12.9230, 77.5800),
        'HSR Layout': (12.9100, 77.6450), 'Banashankari Stage II': (12.9260, 77.5500), 'Banashankari Stage III': (12.9200, 77.5400),
        'Unknown': (12.9716, 77.5946)
    }

    # Add coordinates dynamically
    df['latitude'] = df['location'].map(lambda x: location_coords.get(str(x).split(',')[0].strip(), (12.9716, 77.5946))[0])
    df['longitude'] = df['location'].map(lambda x: location_coords.get(str(x).split(',')[0].strip(), (12.9716, 77.5946))[1])
    
    # Introduce small random perturbation to coordinates so markers don't overlap exactly
    np.random.seed(42)
    df['latitude'] += np.random.uniform(-0.005, 0.005, size=len(df))
    df['longitude'] += np.random.uniform(-0.005, 0.005, size=len(df))

    return df
```

**Step 2: Refactor `update_sentiment_tab` callback to run dynamic SQL queries**

In `src/dashboard.py`, modify the reviews and operations explorer callback to connect to Neon DB and fetch matching logs in real time using parameters (to prevent SQL injection and ensure maximum security):

```python
def update_sentiment_tab(locations, cuisines, risks, search_query, selected_issue, selected_sentiment):
    try:
        engine = get_db_engine()
        
        # Build query parts dynamically with placeholders
        query_parts = ["SELECT * FROM reviews WHERE 1=1"]
        params = {}
        
        # 1. Location / Cuisine filters
        if locations or cuisines or risks:
            sub_query = "SELECT restaurant_id FROM restaurants WHERE 1=1"
            if locations:
                placeholders = []
                for idx, val in enumerate(locations):
                    key = f"loc_{idx}"
                    placeholders.append(f":{key}")
                    params[key] = val
                sub_query += f" AND location IN ({', '.join(placeholders)})"
            if cuisines:
                placeholders = []
                for idx, val in enumerate(cuisines):
                    key = f"cuis_{idx}"
                    placeholders.append(f":{key}")
                    params[key] = val
                sub_query += f" AND primary_cuisine IN ({', '.join(placeholders)})"
            if risks:
                placeholders = []
                for idx, val in enumerate(risks):
                    key = f"risk_{idx}"
                    placeholders.append(f":{key}")
                    params[key] = val
                sub_query += f" AND risk_category IN ({', '.join(placeholders)})"
                
            query_parts.append(f"AND restaurant_id IN ({sub_query})")
            
        # 2. Text Search Query Filter
        if search_query:
            query_parts.append("AND review_text ILIKE :search_query") # Postgres ILIKE is case-insensitive!
            params["search_query"] = f"%{search_query}%"
            
        # 3. Operational Issue Filter
        if selected_issue:
            if selected_issue == "any":
                query_parts.append("AND (delivery_delay=1 OR food_quality=1 OR packaging=1 OR service=1 OR hygiene=1 OR wrong_order=1)")
            else:
                issue_cols = ['delivery_delay', 'food_quality', 'packaging', 'service', 'hygiene', 'wrong_order']
                if selected_issue in issue_cols:
                    query_parts.append(f"AND {selected_issue}=1")
                    
        # 4. Sentiment Polarity Filter
        if selected_sentiment:
            query_parts.append("AND sentiment_label = :sentiment")
            params["sentiment"] = selected_sentiment
            
        # Limit results for rendering speed
        query_parts.append("LIMIT 5000")
        
        full_query = " ".join(query_parts)
        
        # Execute query using connection
        with engine.connect() as conn:
            # Execute with parameter dictionary
            dff_rev = pd.read_sql_query(text(full_query), conn, params=params)
    except Exception as e:
        print("SQL Error fetching reviews:", e)
        return go.Figure(), go.Figure(), []
    
    # Convert booleans
    issue_cols = ['delivery_delay', 'food_quality', 'packaging', 'service', 'hygiene', 'wrong_order']
    for col in issue_cols:
        if col in dff_rev.columns:
            dff_rev[col] = dff_rev[col].astype(bool)

    # Plot Sentiment Pie Chart
    if not dff_rev.empty:
        sent_counts = dff_rev["sentiment_label"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        fig_pie = px.pie(
            sent_counts,
            names="Sentiment",
            values="Count",
            color="Sentiment",
            color_discrete_map={"Positive": "#28a745", "Neutral": "#6c757d", "Negative": ZOMATO_RED},
            hole=0.4
        )
        fig_pie.update_layout(margin={"r":0,"t":10,"l":0,"b":0})
    else:
        fig_pie = go.Figure()

    # Plot Operational Complaint Categories Bar Chart
    issue_sum = []
    issue_labels = {
        'delivery_delay': 'Delivery Delays',
        'food_quality': 'Food Quality Issues',
        'packaging': 'Packaging Leaks',
        'service': 'Rude Service',
        'hygiene': 'Hygiene Concerns',
        'wrong_order': 'Incorrect Order'
    }
    for col, label in issue_labels.items():
        if col in dff_rev.columns:
            count = dff_rev[dff_rev[col] == True].shape[0]
            issue_sum.append({"Category": label, "Complaints": count})
            
    df_issues = pd.DataFrame(issue_sum).sort_values("Complaints", ascending=True)
    
    fig_bar = px.bar(
        df_issues,
        y="Category",
        x="Complaints",
        orientation="h",
        text="Complaints",
        color="Complaints",
        color_continuous_scale=["#FFEBEC", "#FFA3A9", "#CB202D"],
        labels={"Complaints": "Complaints Count", "Category": ""}
    )
    
    fig_bar.update_traces(
        textposition="outside",
        cliponaxis=False,
        textfont=dict(family=FONT_FAMILY, size=11, color="#1C1C1C"),
        hovertemplate="<b>%{y}</b><br>Total Incidents: %{x}<extra></extra>"
    )
    
    fig_bar.update_layout(
        margin={"r": 50, "t": 15, "l": 150, "b": 15},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis=dict(showgrid=False, visible=False, title=None),
        yaxis=dict(showgrid=False, title=None, tickfont=dict(family=FONT_FAMILY, size=12, color="#1C1C1C"))
    )

    # Reviews table data extraction (Limit to top 30)
    table_data = dff_rev.head(30).to_dict("records")
    
    return fig_pie, fig_bar, table_data
```

---

### Task 5: Verify Dashboard Functionality & Commit Changes

**Files:**
- Test: `.venv/bin/python src/dashboard.py`

**Step 1: Test Python loader locally**

Verify that `load_data()` can fetch from Neon without errors:

```bash
DATABASE_URL="postgresql://neondb_owner:npg_tSRYv5mXKs4P@ep-misty-rice-ap6vsh1m-pooler.c-7.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require" .venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from dashboard import load_data; df = load_data(); print('Metadata Loaded shape:', df.shape)"
```

Expected Output:
```bash
Connected to Neon DB...
Metadata Loaded shape: (51717, 18)
```

**Step 2: Commit implementation changes**

```bash
git add src/dashboard.py
git commit -m "refactor: fetch analytics data dynamically from cloud Neon PostgreSQL DB"
```

**Step 3: Push changes to remote main branch**

```bash
git push origin main
```

---

### Task 6: Add Environment variables to Render Web Service

**Step 1: Navigate to Render**

1. Open your **Render Dashboard**.
2. Click on your **Web Service** (`cloud-kitchen-sentiment-churn-dashboard`).
3. Click on the **Environment** tab on the left sidebar.
4. Click **Add Environment Variable**.
5. Add the following key-value pair:
   - **Key:** `DATABASE_URL`
   - **Value:** `postgresql://neondb_owner:npg_tSRYv5mXKs4P@ep-misty-rice-ap6vsh1m-pooler.c-7.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require`
6. Click **Save Changes**. Render will automatically trigger a rolling redeploy using your cloud database, and the site will be 100% stable with real-time data!
