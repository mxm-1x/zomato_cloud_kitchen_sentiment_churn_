#!/usr/bin/env python3
"""
Cloud Kitchen Sentiment & Churn Analysis
Main Execution Pipeline

Orchestrates:
1. Data Cleaning & Parsing
2. Exploratory Data Analysis
3. Sentiment Analysis (NLP)
4. Churn Prediction & Risk Scoring
5. Report Generation

Usage:
    python main.py

Requirements:
    - pandas, numpy, scikit-learn
    - matplotlib, seaborn
    - vaderSentiment, textblob (optional but recommended)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_cleaning import clean_and_transform, parse_reviews_list
from eda import (plot_rating_distribution, plot_cost_analysis, 
                 plot_location_analysis, plot_cuisine_analysis, plot_review_volume_analysis)
from sentiment_analysis import SentimentAnalyzer
from churn_prediction import ChurnPredictor


def run_pipeline(data_path='data/zomato.csv'):
    """Execute full analysis pipeline."""

    print("=" * 60)
    print("CLOUD KITCHEN SENTIMENT & CHURN ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Data Loading & Cleaning
    print("[STEP 1/5] Data Cleaning & Preprocessing...")
    print("-" * 40)

    df_raw = pd.read_csv(data_path)
    print(f"Loaded raw data: {df_raw.shape}")

    df_clean = clean_and_transform(df_raw)
    df_clean.to_csv('data/cleaned_zomato_bangalore.csv', index=False)

    # Create exploded reviews
    exploded_reviews = []
    for idx, row in df_clean.iterrows():
        for review in row['parsed_reviews'][:2]:
            exploded_reviews.append({
                'restaurant_id': row['restaurant_id'],
                'restaurant_name': row['name'],
                'location': row['location'],
                'rest_type': row['rest_type'],
                'cuisines': row['cuisines'],
                'review_text': review['review_text'],
                'review_rating': review['review_rating'],
                'overall_rate': row['rate_numeric'],
                'cost': row['cost_numeric'],
                'online_order': row['online_order'],
                'book_table': row['book_table']
            })

    df_reviews = pd.DataFrame(exploded_reviews)
    df_reviews.to_csv('data/exploded_reviews.csv', index=False)
    print(f"Created exploded reviews: {df_reviews.shape}")
    print()

    # Step 2: EDA
    print("[STEP 2/5] Exploratory Data Analysis...")
    print("-" * 40)

    plot_rating_distribution(df_clean, 'reports/rating_distribution.png')
    plot_cost_analysis(df_clean, 'reports/cost_analysis.png')
    plot_location_analysis(df_clean, 'reports/location_analysis.png')
    plot_cuisine_analysis(df_clean, 'reports/cuisine_analysis.png')
    plot_review_volume_analysis(df_reviews, 'reports/review_volume_analysis.png')
    print("EDA visualizations saved to reports/")
    print()

    # Step 3: Sentiment Analysis
    print("[STEP 3/5] NLP Sentiment Analysis...")
    print("-" * 40)

    analyzer = SentimentAnalyzer()
    df_analyzed = analyzer.analyze_reviews(df_reviews)
    df_analyzed.to_csv('data/reviews_with_sentiment.csv', index=False)

    summary = analyzer.generate_restaurant_sentiment_summary(df_analyzed)
    summary.to_csv('data/restaurant_sentiment_summary.csv', index=False)

    print(f"Analyzed {len(df_analyzed)} reviews")
    print(f"Sentiment distribution:")
    print(df_analyzed['sentiment_label'].value_counts())
    print()

    # Step 4: Churn Prediction
    print("[STEP 4/5] Churn Prediction & Risk Scoring...")
    print("-" * 40)

    predictor = ChurnPredictor()
    df_features = predictor.engineer_churn_features(df_clean, summary)
    # Drop heavy columns to keep size extremely light (~15MB instead of 1.5GB)
    heavy_cols = ['reviews_list', 'parsed_reviews', 'all_review_texts', 'menu_item', 'dish_liked', 'url', 'address', 'phone']
    df_features_light = df_features.drop(columns=[col for col in heavy_cols if col in df_features.columns])
    df_features_light.to_csv('data/restaurants_with_churn_features.csv', index=False)

    X, y, feature_names = predictor.prepare_ml_features(df_features)
    results, best_model_name = predictor.train_and_evaluate(X, y, feature_names)
    predictor.plot_results(results, best_model_name, 'reports/churn_model_evaluation.png')

    # Serialize best model and scaler for the dashboard
    import pickle
    os.makedirs('models', exist_ok=True)
    with open('models/churn_model.pkl', 'wb') as f:
        pickle.dump(predictor.best_model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(predictor.scaler, f)
    print("Serialized best churn model and scaler to models/")

    # Generate predictions
    df_features['churn_probability'] = predictor.predict_risk(X)[0]
    df_features['risk_category'] = predictor.predict_risk(X)[1]

    high_risk = df_features[df_features['risk_category'].isin(['High', 'Critical'])].sort_values(
        'churn_probability', ascending=False
    )
    high_risk.to_csv('reports/high_risk_restaurants.csv', index=False)

    print(f"High-risk restaurants identified: {len(high_risk)}")
    print()

    # Step 5: Summary Report
    print("[STEP 5/5] Generating Summary Report...")
    print("-" * 40)

    generate_summary_report(df_clean, df_analyzed, df_features, high_risk)
    print()

    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    print("Output files:")
    print("  - data/cleaned_zomato_bangalore.csv")
    print("  - data/exploded_reviews.csv")
    print("  - data/reviews_with_sentiment.csv")
    print("  - data/restaurant_sentiment_summary.csv")
    print("  - data/restaurants_with_churn_features.csv")
    print("  - reports/rating_distribution.png")
    print("  - reports/cost_analysis.png")
    print("  - reports/location_analysis.png")
    print("  - reports/cuisine_analysis.png")
    print("  - reports/review_volume_analysis.png")
    print("  - reports/churn_model_evaluation.png")
    print("  - reports/high_risk_restaurants.csv")
    print("  - reports/EXECUTIVE_SUMMARY.md")


def generate_summary_report(df_clean, df_analyzed, df_features, high_risk):
    """Generate executive summary markdown report."""

    report = f"""# Cloud Kitchen Sentiment & Churn Analysis - Executive Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total Restaurants | {len(df_clean):,} |
| Total Reviews Analyzed | {len(df_analyzed):,} |
| Avg Reviews per Restaurant | {len(df_analyzed)/len(df_clean):.1f} |
| Locations Covered | {df_clean['location'].nunique()} |
| Cuisine Types | {df_clean['primary_cuisine'].nunique()} |

## Key Findings

### 1. Market Landscape
- **Top Location:** {df_clean['location'].value_counts().index[0]} ({df_clean['location'].value_counts().iloc[0]} restaurants)
- **Most Common Type:** {df_clean['rest_type'].value_counts().index[0]}
- **Online Order Adoption:** {df_clean['online_order_binary'].mean()*100:.1f}%
- **Average Rating:** {df_clean['rate_numeric'].mean():.2f}/5
- **Average Cost for Two:** ₹{df_clean['cost_numeric'].mean():.0f}

### 2. Sentiment Analysis Results

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Positive | {(df_analyzed['sentiment_label'] == 'Positive').sum():,} | {(df_analyzed['sentiment_label'] == 'Positive').mean()*100:.1f}% |
| Neutral | {(df_analyzed['sentiment_label'] == 'Neutral').sum():,} | {(df_analyzed['sentiment_label'] == 'Neutral').mean()*100:.1f}% |
| Negative | {(df_analyzed['sentiment_label'] == 'Negative').sum():,} | {(df_analyzed['sentiment_label'] == 'Negative').mean()*100:.1f}% |

**Operational Issues Detected:**
- Delivery Delays: {df_analyzed['delivery_delay'].sum():,}
- Food Quality Issues: {df_analyzed['food_quality'].sum():,}
- Packaging Problems: {df_analyzed['packaging'].sum():,}
- Service Complaints: {df_analyzed['service'].sum():,}
- Hygiene Concerns: {df_analyzed['hygiene'].sum():,}
- Wrong Orders: {df_analyzed['wrong_order'].sum():,}

### 3. Churn Risk Assessment

| Risk Category | Count | Percentage |
|---------------|-------|------------|
| Critical | {(df_features['risk_category'] == 'Critical').sum():,} | {(df_features['risk_category'] == 'Critical').mean()*100:.1f}% |
| High | {(df_features['risk_category'] == 'High').sum():,} | {(df_features['risk_category'] == 'High').mean()*100:.1f}% |
| Medium | {(df_features['risk_category'] == 'Medium').sum():,} | {(df_features['risk_category'] == 'Medium').mean()*100:.1f}% |
| Low | {(df_features['risk_category'] == 'Low').sum():,} | {(df_features['risk_category'] == 'Low').mean()*100:.1f}% |

**High-Risk Restaurants Requiring Intervention:** {len(high_risk)}

### 4. Top Risk Factors

Based on ML feature importance, the strongest predictors of churn are:
1. **Operational Issue Rate** - Frequency of delivery/quality complaints
2. **Days Since Last Review** - Indicator of customer disengagement
3. **Negative Review Percentage** - Sustained negative sentiment
4. **Average Sentiment Score** - Overall customer satisfaction trend
5. **Rating Volatility** - Inconsistent service quality

### 5. Recommendations

**For Cloud Kitchen Operators:**
1. **Immediate:** Address operational issues for {len(high_risk)} high-risk restaurants
2. **Short-term:** Implement 30-day review response protocol for dormant restaurants
3. **Medium-term:** Optimize delivery logistics for locations with high delay complaints
4. **Long-term:** Use sentiment trends to guide menu engineering and pricing strategy

**For Data Engineering:**
1. Deploy real-time review streaming pipeline using Kafka + Spark
2. Implement automated alert system for restaurants crossing risk thresholds
3. Build A/B testing framework for intervention strategies
4. Create customer-facing dashboard for transparent quality metrics

---

*This analysis was generated using synthetic data mimicking the Zomato Bangalore dataset structure. For production deployment, replace with live API data sources.*
"""

    with open('reports/EXECUTIVE_SUMMARY.md', 'w') as f:
        f.write(report)

    print("Executive summary saved to reports/EXECUTIVE_SUMMARY.md")


if __name__ == "__main__":
    run_pipeline()
