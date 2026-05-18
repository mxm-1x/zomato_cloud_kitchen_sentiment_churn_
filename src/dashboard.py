import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
import os
# Load local .env file if it exists
if os.path.exists('.env'):
    with open('.env') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ[k] = v.strip('"\'')

import plotly.io as pio


# Global font
FONT_FAMILY = "'Outfit', sans-serif"

pio.templates["zomato_glass"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT_FAMILY, color="#1C1C1C"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        colorway=["#CB202D", "#E4B02E", "#1C1C1C", "#6B7280"]
    )
)
pio.templates.default = "zomato_glass"

# Initialize Dash App with flatly bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap"
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "Zomato Bangalore Cloud Kitchen Analytics"
server = app.server

# -----------------------------------------------------------------------------
# ZOMATO COLOR PALETTE & HIGH-END DESIGN SYSTEM
# -----------------------------------------------------------------------------
ZOMATO_RED = "#CB202D"        # Premium Brand Red
ZOMATO_DARK = "#1C1C1C"       # Text dark
ZOMATO_LIGHT_BG = "#F4F5F7"   # Sleek cool-gray app Background
ZOMATO_WHITE = "rgba(255, 255, 255, 0.95)" # Glassmorphic white
ZOMATO_ACCENT = "#E4B02E"     # Warm accent
ZOMATO_GRAY = "#6B7280"       # Modern gray text

# Global font
FONT_FAMILY = "'Outfit', sans-serif"

# Custom component styles
HEADER_STYLE = {
    "background": f"linear-gradient(135deg, {ZOMATO_RED} 0%, #A31823 100%)",
    "color": "#FFFFFF",
    "padding": "24px 30px",
    "borderRadius": "0 0 24px 24px",
    "boxShadow": "0 10px 30px rgba(203, 32, 45, 0.3)",
    "marginBottom": "30px",
    "fontFamily": FONT_FAMILY
}

SIDEBAR_STYLE = {
    "backgroundColor": ZOMATO_WHITE,
    "backdropFilter": "blur(12px)",
    "padding": "25px",
    "borderRadius": "16px",
    "border": "1px solid rgba(255, 255, 255, 0.5)",
    "boxShadow": "0 8px 32px rgba(0, 0, 0, 0.04)",
    "marginBottom": "20px",
    "fontFamily": FONT_FAMILY
}

CARD_STYLE = {
    "backgroundColor": ZOMATO_WHITE,
    "backdropFilter": "blur(12px)",
    "borderRadius": "16px",
    "border": "1px solid rgba(255, 255, 255, 0.6)",
    "boxShadow": "0 8px 32px rgba(31, 38, 135, 0.05)",
    "padding": "24px",
    "marginBottom": "24px",
    "transition": "transform 0.3s ease, box-shadow 0.3s ease",
    "fontFamily": FONT_FAMILY
}

KPI_CARD_STYLE = {
    "backgroundColor": ZOMATO_WHITE,
    "backdropFilter": "blur(12px)",
    "borderRadius": "16px",
    "border": "1px solid rgba(255, 255, 255, 0.6)",
    "borderLeft": f"6px solid {ZOMATO_RED}",
    "boxShadow": "0 8px 32px rgba(31, 38, 135, 0.05)",
    "padding": "24px",
    "textAlign": "center",
    "fontFamily": FONT_FAMILY,
    "transition": "transform 0.2s ease"
}

# -----------------------------------------------------------------------------
# DATA LOADING & INITIALIZATION
# -----------------------------------------------------------------------------
from sqlalchemy import create_engine, text

def get_db_engine():
    # Read from environment, fallback to Neon connection string
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
        # Fetch restaurants metadata from Postgres (only loads the 16 columns we actually use, reducing WAN overhead)
        columns_to_load = [
            'restaurant_id', 'name', 'location', 'rate_numeric', 'cost_numeric',
            'votes_numeric', 'primary_cuisine', 'online_order_binary',
            'book_table_binary', 'review_count', 'avg_sentiment',
            'negative_review_pct', 'operational_issue_rate',
            'days_since_last_review', 'churn_probability', 'risk_category'
        ]
        query = f"SELECT {', '.join(columns_to_load)} FROM restaurants"
        df = pd.read_sql_query(query, engine)
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

    return df, pd.DataFrame()

# Load datasets
df_restaurants, df_reviews = load_data()

# Clean drop downs
locations = sorted([str(x) for x in df_restaurants['location'].dropna().unique()])
cuisines = sorted([str(x) for x in df_restaurants['primary_cuisine'].dropna().unique()])

# Load ML Churn Model and Scaler
try:
    with open('models/churn_model.pkl', 'rb') as f:
        churn_model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    ML_AVAILABLE = True
except Exception:
    churn_model = None
    scaler = None
    ML_AVAILABLE = False
    print("Pre-trained ML Model not found. Churn simulator will run on fallback business rule model.")

# -----------------------------------------------------------------------------
# APP LAYOUT
# -----------------------------------------------------------------------------
app.layout = html.Div(
    style={"backgroundColor": ZOMATO_LIGHT_BG, "minHeight": "100vh"},
    children=[
        # 1. Zomato Crimson Header Navbar
        html.Div(
            style=HEADER_STYLE,
            className="d-flex justify-content-between align-items-center flex-wrap",
            children=[
                html.Div(
                    children=[
                        html.H1(
                            "zomato",
                            style={
                                "fontFamily": "'Outfit', 'Helvetica Neue', sans-serif",
                                "fontWeight": "900",
                                "fontStyle": "italic",
                                "fontSize": "2.8rem",
                                "margin": "0",
                                "display": "inline-block",
                                "letterSpacing": "-1.5px"
                            }
                        ),
                        html.Span(
                            " Bangalore Cloud Kitchen Performance & Churn Analytics",
                            style={"fontSize": "1.3rem", "fontWeight": "300", "marginLeft": "15px"}
                        )
                    ]
                ),
                html.Div(
                    children=[
                        dbc.Badge(
                            "ML Model Active" if ML_AVAILABLE else "Rule Model Active",
                            color="success" if ML_AVAILABLE else "warning",
                            className="p-2",
                            style={"fontSize": "0.9rem", "borderRadius": "20px"}
                        )
                    ]
                )
            ]
        ),

        # 2. Main Body Container
        dbc.Container(
            fluid=True,
            children=[
                dbc.Row([
                    # Sidebar Controls (Column 3)
                    dbc.Col(
                        width=3,
                        children=[
                            html.Div(
                                style=SIDEBAR_STYLE,
                                children=[
                                    html.H4("Filter Panel", style={"fontWeight": "700", "color": ZOMATO_DARK, "marginBottom": "20px"}),
                                    
                                    # Location Selector
                                    html.Div([
                                        html.Label("Select Location(s)", style={"fontWeight": "600", "color": ZOMATO_DARK}),
                                        dcc.Dropdown(
                                            id="location-filter",
                                            options=[{"label": loc, "value": loc} for loc in locations],
                                            value=[],
                                            multi=True,
                                            placeholder="All Locations",
                                            style={"marginBottom": "20px"}
                                        )
                                    ]),

                                    # Cuisine Selector
                                    html.Div([
                                        html.Label("Select Cuisine(s)", style={"fontWeight": "600", "color": ZOMATO_DARK}),
                                        dcc.Dropdown(
                                            id="cuisine-filter",
                                            options=[{"label": cuis, "value": cuis} for cuis in cuisines],
                                            value=[],
                                            multi=True,
                                            placeholder="All Cuisines",
                                            style={"marginBottom": "20px"}
                                        )
                                    ]),

                                    # Risk Level Selector
                                    html.Div([
                                        html.Label("Churn Risk Category", style={"fontWeight": "600", "color": ZOMATO_DARK}),
                                        dcc.Dropdown(
                                            id="risk-filter",
                                            options=[
                                                {"label": "Critical", "value": "Critical"},
                                                {"label": "High", "value": "High"},
                                                {"label": "Medium", "value": "Medium"},
                                                {"label": "Low", "value": "Low"}
                                            ],
                                            value=[],
                                            multi=True,
                                            placeholder="All Risk Levels",
                                            style={"marginBottom": "20px"}
                                        )
                                    ]),

                                    # Reset Button
                                    dbc.Button(
                                        "Clear Filters",
                                        id="clear-btn",
                                        color="outline-secondary",
                                        size="sm",
                                        className="w-100",
                                        style={"borderRadius": "8px", "fontWeight": "600"}
                                    )
                                ]
                            ),
                            
                            # Brand Promotion Footer
                            html.Div(
                                style={
                                    "textAlign": "center",
                                    "color": ZOMATO_GRAY,
                                    "fontSize": "0.8rem",
                                    "marginTop": "20px"
                                },
                                children=[
                                    html.P("Zomato Cloud Kitchen Analytics Platform v1.0"),
                                    html.P("© 2026 Zomato Bangalore Ops Team")
                                ]
                            )
                        ]
                    ),

                    # Dashboard Content Area (Column 9)
                    dbc.Col(
                        width=9,
                        children=[
                            # Tabs
                            dbc.Tabs(
                                id="tabs",
                                active_tab="overview-tab",
                                style={"marginBottom": "20px"},
                                children=[
                                    # Tab 1: Overview
                                    dbc.Tab(
                                        label="Business & Market Overview",
                                        tab_id="overview-tab",
                                        label_style={"fontWeight": "700", "color": ZOMATO_DARK},
                                        active_label_style={"color": ZOMATO_RED, "borderBottom": f"3px solid {ZOMATO_RED}"},
                                        children=[
                                            html.Div(style={"marginTop": "15px"}, children=[
                                                # Row 1: KPI Cards
                                                dbc.Row([
                                                    dbc.Col(dbc.Card(id="kpi-kitchens", style=KPI_CARD_STYLE), width=3),
                                                    dbc.Col(dbc.Card(id="kpi-rating", style=KPI_CARD_STYLE), width=3),
                                                    dbc.Col(dbc.Card(id="kpi-risk", style=KPI_CARD_STYLE), width=3),
                                                    dbc.Col(dbc.Card(id="kpi-cost", style=KPI_CARD_STYLE), width=3),
                                                ], className="g-3 mb-4"),

                                                # Row 2: Maps and Distribution
                                                dbc.Row([
                                                    # Bangalore Cloud Kitchen Hotspots
                                                    dbc.Col(
                                                        dbc.Card(
                                                            style=CARD_STYLE,
                                                            children=[
                                                                html.H5("Bangalore Cloud Kitchen Hotspots", style={"fontWeight": "700", "color": ZOMATO_DARK}),
                                                                html.P("Geographic density and churn risk overlay", style={"fontSize": "0.85rem", "color": ZOMATO_GRAY}),
                                                                dcc.Graph(id="bangalore-map", style={"height": "400px"})
                                                            ]
                                                        ),
                                                        width=7
                                                    ),
                                                    # Rating Trajectory / Cost Breakdown
                                                    dbc.Col(
                                                        dbc.Card(
                                                            style=CARD_STYLE,
                                                            children=[
                                                                html.H5("Rating vs. Cost Dynamics", style={"fontWeight": "700", "color": ZOMATO_DARK}),
                                                                html.P("Operational price efficiency by rating bracket", style={"fontSize": "0.85rem", "color": ZOMATO_GRAY}),
                                                                dcc.Graph(id="cost-rating-scatter", style={"height": "400px"})
                                                            ]
                                                        ),
                                                        width=5
                                                    )
                                                ], className="g-3 mb-4"),

                                                # Row 3: Bottom Analytics
                                                dbc.Row([
                                                    # High Risk Leaders
                                                    dbc.Col(
                                                        dbc.Card(
                                                            style=CARD_STYLE,
                                                            children=[
                                                                html.H5("Most Competed Neighborhoods", style={"fontWeight": "700", "color": ZOMATO_DARK}),
                                                                dcc.Graph(id="location-competitiveness-chart")
                                                            ]
                                                        ),
                                                        width=6
                                                    ),
                                                    # Cuisine breakdowns
                                                    dbc.Col(
                                                        dbc.Card(
                                                            style=CARD_STYLE,
                                                            children=[
                                                                html.H5("Primary Cuisine Churn Vulnerability", style={"fontWeight": "700", "color": ZOMATO_DARK}),
                                                                dcc.Graph(id="cuisine-vulnerability-chart")
                                                            ]
                                                        ),
                                                        width=6
                                                    )
                                                ], className="g-3")
                                            ])
                                        ]
                                    ),

                                    # Tab 2: Sentiment & Operations
                                    dbc.Tab(
                                        label="Sentiment & Operations",
                                        tab_id="sentiment-tab",
                                        label_style={"fontWeight": "700", "color": ZOMATO_DARK},
                                        active_label_style={"color": ZOMATO_RED, "borderBottom": f"3px solid {ZOMATO_RED}"},
                                        children=[
                                            html.Div(style={"marginTop": "15px"}, children=[
                                                # Row 1: Sentiment Overview Graphs
                                                dbc.Row([
                                                    # Sentiment Polarity Distribution
                                                    dbc.Col(
                                                        dbc.Card(
                                                            style=CARD_STYLE,
                                                            children=[
                                                                html.H5("Overall Review Sentiment Distribution", style={"fontWeight": "700"}),
                                                                dcc.Graph(id="sentiment-pie-chart", style={"height": "320px"})
                                                            ]
                                                        ),
                                                        width=5
                                                    ),
                                                    # Operational Complaint Category Volumes
                                                    dbc.Col(
                                                        dbc.Card(
                                                            style=CARD_STYLE,
                                                            children=[
                                                                html.H5("Operational Failure Incident Volumes", style={"fontWeight": "700"}),
                                                                dcc.Graph(id="operational-bar-chart", style={"height": "320px"})
                                                            ]
                                                        ),
                                                        width=7
                                                    )
                                                ], className="g-3 mb-4"),

                                                # Row 2: Customer Review Search table
                                                dbc.Row([
                                                    dbc.Col(
                                                        dbc.Card(
                                                            style=CARD_STYLE,
                                                            children=[
                                                                html.H5("Operational Review Feed Explorer", style={"fontWeight": "700", "marginBottom": "15px"}),
                                                                
                                                                # Interactive filters for review table
                                                                dbc.Row([
                                                                    dbc.Col([
                                                                        html.Label("Search Review Content:", style={"fontSize": "0.85rem", "fontWeight": "600"}),
                                                                        dbc.Input(id="search-input", placeholder="e.g. cold, late, hair...", type="text", size="sm")
                                                                    ], width=4),
                                                                    
                                                                    dbc.Col([
                                                                        html.Label("Filter Operational Issue:", style={"fontSize": "0.85rem", "fontWeight": "600"}),
                                                                        dcc.Dropdown(
                                                                            id="issue-select",
                                                                            options=[
                                                                                {"label": "Any Operational Failures", "value": "any"},
                                                                                {"label": "Delivery Delay", "value": "delivery_delay"},
                                                                                {"label": "Food Quality Issue", "value": "food_quality"},
                                                                                {"label": "Packaging Leak/Mess", "value": "packaging"},
                                                                                {"label": "Rude Service", "value": "service"},
                                                                                {"label": "Hygiene/Dirty", "value": "hygiene"},
                                                                                {"label": "Incorrect/Wrong Order", "value": "wrong_order"}
                                                                            ],
                                                                            value=None,
                                                                            placeholder="Select Operational Issue",
                                                                            style={"fontSize": "0.9rem"}
                                                                        )
                                                                    ], width=4),
                                                                    
                                                                    dbc.Col([
                                                                        html.Label("Filter Sentiment Polarity:", style={"fontSize": "0.85rem", "fontWeight": "600"}),
                                                                        dcc.Dropdown(
                                                                            id="sentiment-select",
                                                                            options=[
                                                                                {"label": "Positive Only", "value": "Positive"},
                                                                                {"label": "Neutral Only", "value": "Neutral"},
                                                                                {"label": "Negative Only", "value": "Negative"}
                                                                            ],
                                                                            value=None,
                                                                            placeholder="Select Sentiment"
                                                                        )
                                                                    ], width=4)
                                                                ], className="mb-3 g-2"),

                                                                # Reviews DataTable
                                                                dash_table.DataTable(
                                                                    id="reviews-table",
                                                                    columns=[
                                                                        {"name": "Restaurant", "id": "restaurant_name"},
                                                                        {"name": "Rating", "id": "review_rating"},
                                                                        {"name": "Sentiment", "id": "sentiment_label"},
                                                                        {"name": "Customer Review", "id": "review_text"}
                                                                    ],
                                                                    page_size=5,
                                                                    style_table={"overflowX": "auto"},
                                                                    style_header={
                                                                        "backgroundColor": ZOMATO_RED,
                                                                        "color": ZOMATO_WHITE,
                                                                        "fontWeight": "bold",
                                                                        "textAlign": "left"
                                                                    },
                                                                    style_cell={
                                                                        "padding": "10px",
                                                                        "fontFamily": "sans-serif",
                                                                        "fontSize": "0.85rem",
                                                                        "minWidth": "100px",
                                                                        "maxWidth": "400px",
                                                                        "textOverflow": "ellipsis",
                                                                        "overflow": "hidden"
                                                                    },
                                                                    style_data_conditional=[
                                                                        {
                                                                            "if": {"row_index": "odd"},
                                                                            "backgroundColor": "#FAF5F5" # Very light pinkish-gray alternate row
                                                                        },
                                                                        {
                                                                            "if": {"column_id": "sentiment_label", "filter_query": "{sentiment_label} eq 'Positive'"},
                                                                            "color": "#28a745", "fontWeight": "bold"
                                                                        },
                                                                        {
                                                                            "if": {"column_id": "sentiment_label", "filter_query": "{sentiment_label} eq 'Negative'"},
                                                                            "color": ZOMATO_RED, "fontWeight": "bold"
                                                                        }
                                                                    ]
                                                                )
                                                            ]
                                                        ),
                                                        width=12
                                                    )
                                                ], className="g-3")
                                            ])
                                        ]
                                    ),

                                    # Tab 3: Churn Risk Simulator (ML Playground)
                                    dbc.Tab(
                                        label="Risk Simulator (ML Sandbox)",
                                        tab_id="simulator-tab",
                                        label_style={"fontWeight": "700", "color": ZOMATO_DARK},
                                        active_label_style={"color": ZOMATO_RED, "borderBottom": f"3px solid {ZOMATO_RED}"},
                                        children=[
                                            html.Div(style={"marginTop": "15px"}, children=[
                                                dbc.Row([
                                                    # Simulator Sliders Panel
                                                    dbc.Col(
                                                        dbc.Card(
                                                            style=CARD_STYLE,
                                                            children=[
                                                                html.H5("Kitchen Parameters", style={"fontWeight": "700", "marginBottom": "20px"}),
                                                                
                                                                # Slider 1: Average rating
                                                                html.Div([
                                                                    html.Label("Overall Rating (rate_numeric):", style={"fontWeight": "600"}),
                                                                    dcc.Slider(
                                                                        id="sim-rating", min=1.0, max=5.0, step=0.1, value=3.8,
                                                                        marks={i: str(i) for i in range(1, 6)},
                                                                        tooltip={"placement": "bottom", "always_visible": True}
                                                                    )
                                                                ], className="mb-4"),

                                                                # Slider 2: Approx Cost
                                                                html.Div([
                                                                    html.Label("Approx Cost for Two (cost_numeric):", style={"fontWeight": "600"}),
                                                                    dcc.Slider(
                                                                        id="sim-cost", min=100, max=2000, step=50, value=500,
                                                                        marks={100: "100", 500: "500", 1000: "1000", 1500: "1500", 2000: "2000"},
                                                                        tooltip={"placement": "bottom", "always_visible": True}
                                                                    )
                                                                ], className="mb-4"),

                                                                # Slider 3: Review count
                                                                html.Div([
                                                                    html.Label("Total Customer Reviews (review_count):", style={"fontWeight": "600"}),
                                                                    dcc.Slider(
                                                                        id="sim-reviews", min=0, max=100, step=1, value=15,
                                                                        marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
                                                                        tooltip={"placement": "bottom", "always_visible": True}
                                                                    )
                                                                ], className="mb-4"),

                                                                # Slider 4: Operational Failure Rate
                                                                html.Div([
                                                                    html.Label("Operational Failure Rate (% of reviews with complaints):", style={"fontWeight": "600"}),
                                                                    dcc.Slider(
                                                                        id="sim-failure-rate", min=0, max=100, step=5, value=10,
                                                                        marks={i: f"{i}%" for i in range(0, 101, 20)},
                                                                        tooltip={"placement": "bottom", "always_visible": True}
                                                                    )
                                                                ], className="mb-4"),

                                                                # Slider 5: Days since last review
                                                                html.Div([
                                                                    html.Label("Days Since Last Review (Dormancy):", style={"fontWeight": "600"}),
                                                                    dcc.Slider(
                                                                        id="sim-dormancy", min=0, max=365, step=5, value=20,
                                                                        marks={0: "0d", 90: "90d", 180: "180d", 270: "270d", 365: "365d"},
                                                                        tooltip={"placement": "bottom", "always_visible": True}
                                                                    )
                                                                ], className="mb-4"),
                                                                
                                                                dbc.Alert(
                                                                    "Tweak parameters above to see the real-time calculated churn risk probability based on our Machine Learning Churn prediction logic.",
                                                                    color="info", className="mt-3", style={"fontSize": "0.85rem"}
                                                                )
                                                            ]
                                                        ),
                                                        width=7
                                                    ),

                                                    # Prediction Output Dashboard
                                                    dbc.Col(
                                                        children=[
                                                            # Gauge Predictor Output
                                                            dbc.Card(
                                                                style=CARD_STYLE,
                                                                children=[
                                                                    html.H5("Predicted Churn Probability", style={"fontWeight": "700", "textAlign": "center"}),
                                                                    dcc.Graph(id="churn-gauge", style={"height": "320px"}),
                                                                    html.Div(
                                                                        id="risk-category-badge",
                                                                        style={
                                                                            "textAlign": "center",
                                                                            "marginTop": "-20px",
                                                                            "fontSize": "1.3rem",
                                                                            "fontWeight": "bold"
                                                                        }
                                                                    )
                                                                ]
                                                            ),

                                                            # Operational Insights Card
                                                            dbc.Card(
                                                                style=CARD_STYLE,
                                                                children=[
                                                                    html.H5("Model Prediction Details", style={"fontWeight": "700"}),
                                                                    html.Div(id="simulator-details")
                                                                ]
                                                            )
                                                        ],
                                                        width=5
                                                    )
                                                ], className="g-3")
                                            ])
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ])
            ]
        )
    ]
)

# -----------------------------------------------------------------------------
# CALLBACK 1: CLEAR ALL FILTERS
# -----------------------------------------------------------------------------
@app.callback(
    [Output("location-filter", "value"),
     Output("cuisine-filter", "value"),
     Output("risk-filter", "value")],
    [Input("clear-btn", "n_clicks")],
    prevent_initial_call=True
)
def clear_all_filters(n_clicks):
    return [], [], []

# -----------------------------------------------------------------------------
# CALLBACK 2: UPDATE BUSINESS OVERVIEW TAB
# -----------------------------------------------------------------------------
@app.callback(
    [Output("kpi-kitchens", "children"),
     Output("kpi-rating", "children"),
     Output("kpi-risk", "children"),
     Output("kpi-cost", "children"),
     Output("bangalore-map", "figure"),
     Output("cost-rating-scatter", "figure"),
     Output("location-competitiveness-chart", "figure"),
     Output("cuisine-vulnerability-chart", "figure")],
    [Input("location-filter", "value"),
     Input("cuisine-filter", "value"),
     Input("risk-filter", "value")]
)
def update_overview_tab(selected_locations, selected_cuisines, selected_risks):
    # Filter dataset
    dff = df_restaurants.copy()
    if selected_locations:
        dff = dff[dff["location"].isin(selected_locations)]
    if selected_cuisines:
        dff = dff[dff["primary_cuisine"].isin(selected_cuisines)]
    if selected_risks:
        dff = dff[dff["risk_category"].isin(selected_risks)]

    # 1. Compute KPIs
    total_kitchens = len(dff)
    
    avg_rating = dff["rate_numeric"].mean()
    avg_rating_text = f"{avg_rating:.2f} / 5" if not pd.isna(avg_rating) else "N/A"
    
    critical_count = dff[dff["risk_category"].isin(["High", "Critical"])].shape[0]
    critical_pct = (critical_count / total_kitchens * 100) if total_kitchens > 0 else 0
    
    avg_cost = dff["cost_numeric"].mean()
    avg_cost_text = f"₹{avg_cost:.0f}" if not pd.isna(avg_cost) else "N/A"

    # KPI Layout formatting
    kpis = [
        # Total kitchens
        [html.H6("Total Cloud Kitchens", style={"color": ZOMATO_GRAY, "fontSize": "0.9rem"}), 
         html.H2(f"{total_kitchens:,}", style={"color": ZOMATO_RED, "fontWeight": "bold"})],
        # Avg rating
        [html.H6("Average Rating", style={"color": ZOMATO_GRAY, "fontSize": "0.9rem"}), 
         html.H2(avg_rating_text, style={"color": ZOMATO_DARK, "fontWeight": "bold"})],
        # High Risk Count
        [html.H6("High/Critical Churn Risk", style={"color": ZOMATO_GRAY, "fontSize": "0.9rem"}), 
         html.H2(f"{critical_count:,} ({critical_pct:.1f}%)", style={"color": ZOMATO_RED if critical_pct > 20 else ZOMATO_DARK, "fontWeight": "bold"})],
        # Cost for two
        [html.H6("Average Cost (for two)", style={"color": ZOMATO_GRAY, "fontSize": "0.9rem"}), 
         html.H2(avg_cost_text, style={"color": ZOMATO_DARK, "fontWeight": "bold"})]
    ]

    # 2. Bangalore Geographic Plot
    if not dff.empty:
        fig_map = px.scatter_mapbox(
            dff,
            lat="latitude",
            lon="longitude",
            color="risk_category",
            size=dff["votes_numeric"].fillna(0) + 1,
            hover_name="name",
            hover_data={"location": True, "rate_numeric": True, "cost_numeric": True, "churn_probability": ":.2f", "votes_numeric": True, "risk_category": False},
            color_discrete_map={"Critical": "#D9534F", "High": "#F0AD4E", "Medium": "#5BC0DE", "Low": "#5CB85C"},
            zoom=10.5,
            center={"lat": 12.94, "lon": 77.625}
        )
        fig_map.update_layout(
            mapbox_style="carto-positron",
            margin={"r":0,"t":0,"l":0,"b":0},
            legend_title="Risk Profile",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
    else:
        fig_map = go.Figure()
        fig_map.update_layout(title="No kitchens match standard filters.")

    # 3. Cost vs. Rating Scatter
    if not dff.empty:
        fig_scatter = px.scatter(
            dff,
            x="cost_numeric",
            y="rate_numeric",
            color="risk_category",
            hover_name="name",
            size="review_count",
            labels={"cost_numeric": "Cost for Two (₹)", "rate_numeric": "Restaurant Rating", "risk_category": "Risk Level"},
            color_discrete_map={"Critical": "#D9534F", "High": "#F0AD4E", "Medium": "#5BC0DE", "Low": "#5CB85C"}
        )
        fig_scatter.update_layout(
            margin={"r":10,"t":20,"l":10,"b":10},
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#EDEDED"),
            yaxis=dict(showgrid=True, gridcolor="#EDEDED"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    else:
        fig_scatter = go.Figure()

    # 4. Location Competitiveness (Top 10 density locations)
    top_locs = dff["location"].value_counts().head(10).reset_index()
    top_locs.columns = ["location", "kitchen_count"]
    fig_loc = px.bar(
        top_locs,
        x="kitchen_count",
        y="location",
        orientation="h",
        color_discrete_sequence=[ZOMATO_RED]
    )
    fig_loc.update_layout(
        margin={"r":10,"t":10,"l":10,"b":10},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#EDEDED", title="Active Cloud Kitchens"),
        yaxis=dict(categoryorder="total ascending", title=None)
    )

    # 5. Cuisine Churn Vulnerability (Average Churn Probability per Cuisine)
    top_cuisines = dff["primary_cuisine"].value_counts().head(15).index
    cuisine_churn = dff[dff["primary_cuisine"].isin(top_cuisines)].groupby("primary_cuisine")["churn_probability"].mean().reset_index()
    cuisine_churn = cuisine_churn.sort_values("churn_probability", ascending=False)
    
    fig_cuis = px.bar(
        cuisine_churn,
        x="primary_cuisine",
        y="churn_probability",
        color="churn_probability",
        color_continuous_scale=["#5CB85C", "#F0AD4E", "#D9534F"],
        labels={"primary_cuisine": "Food Category", "churn_probability": "Average Churn Risk"}
    )
    fig_cuis.update_layout(
        margin={"r":10,"t":10,"l":10,"b":10},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title=None, tickangle=45),
        yaxis=dict(showgrid=True, gridcolor="#EDEDED", tickformat=",.0%"),
        coloraxis_showscale=False
    )

    return kpis[0], kpis[1], kpis[2], kpis[3], fig_map, fig_scatter, fig_loc, fig_cuis


# -----------------------------------------------------------------------------
# CALLBACK 3: UPDATE SENTIMENT & OPERATIONS TAB
# -----------------------------------------------------------------------------
@app.callback(
    [Output("sentiment-pie-chart", "figure"),
     Output("operational-bar-chart", "figure"),
     Output("reviews-table", "data")],
    [Input("location-filter", "value"),
     Input("cuisine-filter", "value"),
     Input("risk-filter", "value"),
     Input("search-input", "value"),
     Input("issue-select", "value"),
     Input("sentiment-select", "value")]
)
def update_sentiment_tab(locations, cuisines, risks, search_query, selected_issue, selected_sentiment):
    try:
        engine = get_db_engine()
        
        # Build query parts dynamically with placeholders
        query_parts = ["SELECT * FROM reviews WHERE 1=1"]
        params = {}
        
        # 1. Location / Cuisine filters mapping to restaurant_id
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
            query_parts.append("AND review_text ILIKE :search_query")
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
    
    # Convert booleans for incident counts
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



# -----------------------------------------------------------------------------
# CALLBACK 4: REAL-TIME ML CHURN RISK SIMULATOR
# -----------------------------------------------------------------------------
@app.callback(
    [Output("churn-gauge", "figure"),
     Output("risk-category-badge", "children"),
     Output("risk-category-badge", "style"),
     Output("simulator-details", "children")],
    [Input("sim-rating", "value"),
     Input("sim-cost", "value"),
     Input("sim-reviews", "value"),
     Input("sim-failure-rate", "value"),
     Input("sim-dormancy", "value")]
)
def update_simulator(rating, cost, reviews, failure_rate, dormancy):
    # Calculate probability
    prob = 0.0
    
    if ML_AVAILABLE and churn_model is not None and scaler is not None:
        try:
            # We must construct a feature vector matching columns of the trained model
            # Base features:
            # 'rate_numeric', 'cost_numeric', 'votes_numeric', 'review_count',
            # 'online_order_binary', 'book_table_binary', 'cuisine_count',
            # 'avg_sentiment', 'negative_review_pct', 'operational_issue_rate',
            # 'rating_volatility', 'cost_efficiency', 'review_engagement',
            # 'location_competition', 'service_score', 'sentiment_rating_gap',
            # 'issue_density', 'days_since_last_review'
            
            # Map simplified simulator sliders to model features
            fail_pct = failure_rate / 100.0
            avg_sentiment = (rating - 3.0) / 2.0  # Linear mapping: 5.0 -> 1.0, 3.0 -> 0.0, 1.0 -> -1.0
            neg_review_pct = max(0.0, min(100.0, (5.0 - rating) * 20.0 + (failure_rate / 2)))
            
            sim_dict = {
                'rate_numeric': rating,
                'cost_numeric': cost,
                'votes_numeric': reviews * 5,  # Approximating votes from review count
                'review_count': reviews,
                'online_order_binary': 1,
                'book_table_binary': 0,
                'cuisine_count': 2,
                'avg_sentiment': avg_sentiment,
                'negative_review_pct': neg_review_pct,
                'operational_issue_rate': fail_pct,
                'rating_volatility': 0.25,
                'cost_efficiency': rating / (cost / 100 + 1),
                'review_engagement': 0.2,
                'location_competition': 50,
                'service_score': 1,
                'sentiment_rating_gap': abs(avg_sentiment * 5 - rating),
                'issue_density': fail_pct,
                'days_since_last_review': dormancy
            }
            
            # The model may contain rest_type or price_category dummies, let's create a DataFrame with 1 row
            # and align columns to exact features trained in our model
            sim_df = pd.DataFrame([sim_dict])
            
            # Fill other dummy variables with 0 to match features
            for col in churn_model.feature_names_in_:
                if col not in sim_df.columns:
                    sim_df[col] = 0
            
            # Reorder columns to match exactly
            sim_df = sim_df[churn_model.feature_names_in_]
            
            # Scaler transform (if standard scalar used on tree models)
            # Standard random forests don't strictly require scaling, but check for coefficients models
            if hasattr(churn_model, 'coef_'):
                X_scaled = scaler.transform(sim_df)
                prob = float(churn_model.predict_proba(X_scaled)[0][1])
            else:
                prob = float(churn_model.predict_proba(sim_df)[0][1])
        except Exception as e:
            print("Simulator ML Error:", e)
            ML_AVAILABLE_LOCAL = False
        else:
            ML_AVAILABLE_LOCAL = True
    else:
        ML_AVAILABLE_LOCAL = False

    # Fallback business rules prediction if ML model is unavailable or throws error
    if not ML_AVAILABLE_LOCAL:
        # Simple logical score (rating, failure rate, and dormancy are primary factors)
        score = 0.0
        if rating < 3.2:
            score += 0.3
        if rating < 2.5:
            score += 0.2
        if failure_rate > 30:
            score += 0.25
        if dormancy > 60:
            score += 0.2
        if dormancy > 120:
            score += 0.15
        if reviews < 3:
            score += 0.1
            
        prob = min(0.99, max(0.01, score))

    # Assign risk category
    if prob >= 0.8:
        risk_cat = "Critical Risk"
        badge_color = ZOMATO_RED
    elif prob >= 0.6:
        risk_cat = "High Risk"
        badge_color = "#FF7A00" # Orange
    elif prob >= 0.3:
        risk_cat = "Medium Risk"
        badge_color = "#FFD200" # Yellow
    else:
        risk_cat = "Low Churn Risk"
        badge_color = "#28A745" # Green

    # 1. Gauge chart formatting
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': "%", 'font': {'size': 44, 'color': ZOMATO_DARK}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': ZOMATO_DARK},
            'bar': {'color': ZOMATO_RED},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#D6D6D6",
            'steps': [
                {'range': [0, 30], 'color': '#ECFBF0'},  # soft green
                {'range': [30, 60], 'color': '#FFFDE7'}, # soft yellow
                {'range': [60, 80], 'color': '#FFF3E0'}, # soft orange
                {'range': [80, 100], 'color': '#FFEBEE'} # soft red
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig_gauge.update_layout(
        margin={"r":20,"t":40,"l":20,"b":20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    # 2. Dynamic details text
    details = [
        html.Div([
            html.Strong("Risk Diagnostic Feed:"),
            html.Ul([
                html.Li(f"Current Rating ({rating:.1f}/5) is {'healthy' if rating >= 3.8 else 'concerning' if rating >= 3.2 else 'critically low and driving immediate churn'}."),
                html.Li(f"Customer Dormancy of {dormancy} days is {'within safe range' if dormancy <= 30 else 'indicative of declining engagement' if dormancy <= 60 else 'highly critical (restaurant is likely inactive)'}."),
                html.Li(f"Operational Complaint density is {failure_rate}% of reviews. {'This indicates perfect logistics!' if failure_rate <= 5 else 'This indicates standard friction.' if failure_rate <= 15 else 'Urgent operational review required! Inefficiencies are heavily impacting customer retention.'}"),
            ], style={"marginTop": "10px"})
        ])
    ]

    badge_style = {
        "textAlign": "center",
        "color": "#FFFFFF",
        "backgroundColor": badge_color,
        "padding": "10px",
        "borderRadius": "8px",
        "boxShadow": "0 4px 10px rgba(0, 0, 0, 0.1)",
        "marginTop": "-10px",
        "width": "60%",
        "margin": "0 auto"
    }

    return fig_gauge, risk_cat, badge_style, details


if __name__ == "__main__":
    # Create the reports folder if not exists
    os.makedirs('reports', exist_ok=True)
    
    # Production-ready execution configuration
    debug_mode = os.environ.get("DASH_DEBUG", "True").lower() == "true"
    host_ip = os.environ.get("DASH_HOST", "0.0.0.0")
    port_num = int(os.environ.get("PORT", 8050))
    
    app.run(debug=debug_mode, host=host_ip, port=port_num)
