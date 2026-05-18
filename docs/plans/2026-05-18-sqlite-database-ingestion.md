# SQLite Database Ingestion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ingest the pre-processed cloud kitchen restaurants and reviews datasets into a structured, indexed SQLite database, and refactor the dashboard callbacks to query this database dynamically, eliminating RAM overhead and memory crashes on Render forever.

**Architecture:** 
- **Database Engine:** A local SQLite database file `data/zomato_analytics.db` which is fully self-contained and committed directly to the Git repository.
- **Ingestion Automation:** A Python script `src/ingest_to_db.py` that processes the CSV datasets, automatically optimizes column datatypes, builds indexes on critical lookup keys (`restaurant_id`, `location`, `primary_cuisine`), and populates `restaurants` and `reviews` tables.
- **Dynamic Callbacks:** Instead of loading and holding all data in RAM at startup, `src/dashboard.py` will only hold the restaurant metadata (~10MB). The heavy NLP reviews explorer will execute targeted, indexed SQL queries dynamically based on dashboard dropdown filters (e.g. `SELECT * FROM reviews WHERE restaurant_id = X LIMIT 500`), resulting in near-zero background memory overhead and instantaneous updates.

**Tech Stack:** SQLite, Python `sqlite3`, `pandas`, Plotly Dash, Gunicorn.

---

### Task 1: Create SQLite Ingestion Script

**Files:**
- Create: `src/ingest_to_db.py`

**Step 1: Write the ingestion script**

Write a clean script that establishes an SQLite connection, parses the CSVs, writes them to SQL, and builds highly optimized indexes:

```python
import os
import sqlite3
import pandas as pd

def ingest_data():
    db_path = 'data/zomato_analytics.db'
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    
    # Remove existing DB if any to start clean
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print("Connecting to SQLite database...")
    conn = sqlite3.connect(db_path)
    
    # 1. Ingest Restaurants
    restaurants_csv = 'data/restaurants_with_churn_features.csv'
    if os.path.exists(restaurants_csv):
        print(f"Reading {restaurants_csv}...")
        df_rest = pd.read_csv(restaurants_csv)
        print("Writing 'restaurants' table to database...")
        df_rest.to_sql('restaurants', conn, index=False, if_exists='replace')
        
        # Create indexes
        cursor = conn.cursor()
        print("Creating index on restaurants(location)...")
        cursor.execute("CREATE INDEX idx_rest_location ON restaurants(location);")
        print("Creating index on restaurants(primary_cuisine)...")
        cursor.execute("CREATE INDEX idx_rest_cuisine ON restaurants(primary_cuisine);")
        print("Creating index on restaurants(restaurant_id)...")
        cursor.execute("CREATE INDEX idx_rest_id ON restaurants(restaurant_id);")
    else:
        print(f"Error: {restaurants_csv} not found!")

    # 2. Ingest Reviews
    reviews_csv = 'data/reviews_with_sentiment.csv'
    if os.path.exists(reviews_csv):
        print(f"Reading {reviews_csv}...")
        # Load columns we need
        reviews_cols = ['restaurant_id', 'restaurant_name', 'review_text', 'review_rating', 'sentiment_label',
                        'delivery_delay', 'food_quality', 'packaging', 'service', 'hygiene', 'wrong_order']
        df_rev = pd.read_csv(reviews_csv)
        # Filter existing columns
        cols_to_load = [c for c in reviews_cols if c in df_rev.columns]
        df_rev = df_rev[cols_to_load]
        
        print("Writing 'reviews' table to database...")
        df_rev.to_sql('reviews', conn, index=False, if_exists='replace')
        
        # Create indexes
        cursor = conn.cursor()
        print("Creating index on reviews(restaurant_id)...")
        cursor.execute("CREATE INDEX idx_rev_rest_id ON reviews(restaurant_id);")
        print("Creating index on reviews(sentiment_label)...")
        cursor.execute("CREATE INDEX idx_rev_sentiment ON reviews(sentiment_label);")
    else:
        print(f"Error: {reviews_csv} not found!")
        
    conn.commit()
    conn.close()
    print("Database ingestion completed successfully!")

if __name__ == '__main__':
    ingest_data()
```

**Step 2: Commit ingestion script**

```bash
git add src/ingest_to_db.py
git commit -m "feat: add SQLite database ingestion script"
```

---

### Task 2: Execute Ingestion Script and Generate the SQLite DB

**Files:**
- Create: `data/zomato_analytics.db`

**Step 1: Execute the ingestion script using the virtual environment**

Run:
```bash
.venv/bin/python src/ingest_to_db.py
```

Expected Output:
```bash
Connecting to SQLite database...
Reading data/restaurants_with_churn_features.csv...
Writing 'restaurants' table to database...
Creating index on restaurants(location)...
Creating index on restaurants(primary_cuisine)...
Creating index on restaurants(restaurant_id)...
Reading data/reviews_with_sentiment.csv...
Writing 'reviews' table to database...
Creating index on reviews(restaurant_id)...
Creating index on reviews(sentiment_label)...
Database ingestion completed successfully!
```

**Step 2: Verify the generated database file size and structure**

Run:
```bash
ls -lh data/zomato_analytics.db
```
Verify that the file is generated and is around ~35MB to ~45MB.

---

### Task 3: Refactor Dashboard to Query SQLite Dynamically

**Files:**
- Modify: `src/dashboard.py`

**Step 1: Modify `load_data()` to query SQLite instead of reading CSVs**

Replace `load_data()` inside `src/dashboard.py` to read the restaurant metadata from the database, and return an empty stub or `None` for reviews (since reviews will be queried dynamically inside callbacks!):

```python
def load_data():
    """Load restaurant metadata from SQLite database with fallbacks."""
    db_path = 'data/zomato_analytics.db'
    
    if os.path.exists(db_path):
        try:
            print("Connecting to SQLite to load restaurant metadata...")
            conn = sqlite3.connect(db_path)
            # Load only restaurants (very fast, loads in ~100ms)
            df = pd.read_sql_query("SELECT * FROM restaurants", conn)
            conn.close()
            print(f"Successfully loaded {len(df)} restaurants from DB.")
        except Exception as e:
            print(f"Error loading from SQLite: {e}. Falling back to CSVs.")
            df = None
    else:
        df = None
        
    # If DB load failed, fallback to CSV loader or Dummy catalog
    if df is None:
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

**Step 2: Refactor the reviews callback (`update_sentiment_tab`) to query SQLite dynamically**

In `src/dashboard.py`, modify the `update_sentiment_tab` callback (around line 860-980) to construct and execute an optimized SQL query against the database instead of using a global `df_reviews` Pandas variable:

```python
# Import sqlite3 at the top if not imported
import sqlite3

# ... (inside update_sentiment_tab callback)
def update_sentiment_tab(locations, cuisines, risks, search_query, selected_issue, selected_sentiment):
    db_path = 'data/zomato_analytics.db'
    
    # If SQLite database is not present, use a safe empty fallback
    if not os.path.exists(db_path):
        return go.Figure(), go.Figure(), []
        
    conn = sqlite3.connect(db_path)
    
    # Build a highly optimized dynamic query
    query_parts = ["SELECT * FROM reviews WHERE 1=1"]
    params = []
    
    # 1. Sidebar Location/Cuisine Filter mapping to individual reviews
    if locations or cuisines or risks:
        sub_query = "SELECT restaurant_id FROM restaurants WHERE 1=1"
        if locations:
            sub_query += f" AND location IN ({','.join(['?']*len(locations))})"
            params.extend(locations)
        if cuisines:
            sub_query += f" AND primary_cuisine IN ({','.join(['?']*len(cuisines))})"
            params.extend(cuisines)
        if risks:
            sub_query += f" AND risk_category IN ({','.join(['?']*len(risks))})"
            params.extend(risks)
            
        query_parts.append(f"AND restaurant_id IN ({sub_query})")
        
    # 2. Text Search Query Filter
    if search_query:
        query_parts.append("AND review_text LIKE ?")
        params.append(f"%{search_query}%")
        
    # 3. Operational Issue Filter
    if selected_issue:
        if selected_issue == "any":
            query_parts.append("AND (delivery_delay=1 OR food_quality=1 OR packaging=1 OR service=1 OR hygiene=1 OR wrong_order=1)")
        else:
            # Escape to be safe, only allow known issues
            issue_cols = ['delivery_delay', 'food_quality', 'packaging', 'service', 'hygiene', 'wrong_order']
            if selected_issue in issue_cols:
                query_parts.append(f"AND {selected_issue}=1")
                
    # 4. Sentiment Polarity Filter
    if selected_sentiment:
        query_parts.append("AND sentiment_label = ?")
        params.append(selected_sentiment)
        
    # Limit results for charts and table display (keeps RAM tiny and rendering ultra-fast!)
    query_parts.append("LIMIT 5000")
    
    full_query = " ".join(query_parts)
    
    # Execute query
    dff_rev = pd.read_sql_query(full_query, conn, params=params)
    conn.close()
    
    # Convert booleans for issues
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

### Task 4: Verify Dashboard Local Functionality

**Files:**
- Test: `.venv/bin/python src/dashboard.py`

**Step 1: Test Python loader integrity**

Run:
```bash
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from dashboard import load_data; df = load_data(); print('Metadata Loaded shape:', df.shape)"
```

Expected Output:
```bash
Metadata Loaded shape: (51717, 18)
```

**Step 2: Commit implementation changes**

```bash
git add src/dashboard.py
git commit -m "refactor: retrieve dashboard data dynamically from indexed SQLite database"
```

---

### Task 5: Deploy SQLite Database to GitHub

**Files:**
- Modify: `.gitignore`
- Create: `data/zomato_analytics.db` (staged)

**Step 1: Stage the SQLite Database file and commit**

We must ensure that the SQLite database file `data/zomato_analytics.db` is NOT ignored by git (it isn't, but let's double check). Add it and commit:

```bash
git add data/zomato_analytics.db
git commit -m "chore: add self-contained compiled SQLite database file"
```

**Step 2: Push commit to remote main branch**

Run:
```bash
git push origin main
```
Verify that the push succeeds. Render will automatically detect this push, rebuild the web app, and be able to read the pre-compiled SQLite database with extremely fast and low-memory performance!
