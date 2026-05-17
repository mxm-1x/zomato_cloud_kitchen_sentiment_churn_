"""
Data Cleaning & Preprocessing Pipeline
Cloud Kitchen Sentiment & Churn Analysis

Handles:
- Parsing reviews_list stringified tuples
- Cleaning ratings, costs, votes
- Handling missing values
- Feature engineering for temporal analysis
"""

import pandas as pd
import numpy as np
import ast
import re
from datetime import datetime, timedelta
import warnings
import random
warnings.filterwarnings('ignore')


def parse_reviews_list(reviews_str):
    """
    Parse the stringified list of tuples from reviews_list column.
    Returns list of (rating_header, review_text, extracted_rating, review_date) tuples.
    """
    if pd.isna(reviews_str) or reviews_str == '[]':
        return []

    try:
        # Use ast.literal_eval to safely parse the stringified list
        reviews = ast.literal_eval(reviews_str)
        parsed = []

        for review in reviews:
            if isinstance(review, (list, tuple)) and len(review) >= 2:
                rating_header = review[0]
                review_text = review[1]

                # Extract numeric rating from header like "Rated 4.0"
                rating_match = re.search(r'Rated\s+([0-9.]+)', str(rating_header))
                extracted_rating = float(rating_match.group(1)) if rating_match else None

                parsed.append({
                    'rating_header': rating_header,
                    'review_text': review_text,
                    'review_rating': extracted_rating
                })
        return parsed
    except (ValueError, SyntaxError):
        return []


def clean_rate_column(rate_str):
    """Convert '4.1/5' or 'NEW' or '-' to numeric rating."""
    if pd.isna(rate_str) or rate_str in ['NEW', '-', 'nan', 'NaN']:
        return np.nan

    # Extract number before /5
    match = re.search(r'([0-9.]+)/5', str(rate_str))
    if match:
        return float(match.group(1))
    return np.nan


def clean_cost_column(cost_str):
    """Remove commas and convert to numeric."""
    if pd.isna(cost_str):
        return np.nan

    cost_clean = re.sub(r'[^0-9]', '', str(cost_str))
    return float(cost_clean) if cost_clean else np.nan


def extract_review_dates(base_date=None, n_reviews=None):
    """Generate synthetic review dates for temporal analysis."""
    if base_date is None:
        base_date = datetime.now()

    dates = []
    for i in range(n_reviews if n_reviews else 10):
        days_ago = random.randint(1, 730)
        dates.append(base_date - timedelta(days=days_ago))
    return dates


def clean_and_transform(df):
    """
    Main cleaning pipeline.

    Args:
        df: Raw Zomato DataFrame

    Returns:
        Cleaned DataFrame with additional engineered features
    """
    df_clean = df.copy()

    print(f"Original shape: {df_clean.shape}")

    # 1. Drop exact duplicates
    df_clean = df_clean.drop_duplicates()
    print(f"After dropping duplicates: {df_clean.shape}")

    # 2. Clean rate column
    df_clean['rate_numeric'] = df_clean['rate'].apply(clean_rate_column)

    # 3. Clean cost column
    df_clean['cost_numeric'] = df_clean['approx_cost(for two people)'].apply(clean_cost_column)

    # 4. Clean votes
    df_clean['votes_numeric'] = pd.to_numeric(df_clean['votes'], errors='coerce')

    # 5. Parse reviews_list
    df_clean['parsed_reviews'] = df_clean['reviews_list'].apply(parse_reviews_list)
    df_clean['review_count'] = df_clean['parsed_reviews'].apply(len)

    # 6. Extract average review rating from parsed reviews
    def avg_review_rating(reviews):
        if not reviews:
            return np.nan
        ratings = [r['review_rating'] for r in reviews if r['review_rating'] is not None]
        return np.mean(ratings) if ratings else np.nan

    df_clean['avg_review_rating'] = df_clean['parsed_reviews'].apply(avg_review_rating)

    # 7. Extract all review texts
    df_clean['all_review_texts'] = df_clean['parsed_reviews'].apply(
        lambda x: ' '.join([r['review_text'] for r in x]) if x else ''
    )

    # 8. Categorical encoding
    df_clean['online_order_binary'] = (df_clean['online_order'] == 'Yes').astype(int)
    df_clean['book_table_binary'] = (df_clean['book_table'] == 'Yes').astype(int)

    # 9. Location and cuisine features
    df_clean['primary_cuisine'] = df_clean['cuisines'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown')
    df_clean['cuisine_count'] = df_clean['cuisines'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

    # 10. Handle missing values
    # Fill rate with avg_review_rating where available
    mask = df_clean['rate_numeric'].isna() & df_clean['avg_review_rating'].notna()
    df_clean.loc[mask, 'rate_numeric'] = df_clean.loc[mask, 'avg_review_rating']

    # Drop rows with no rating and no reviews
    df_clean = df_clean.dropna(subset=['rate_numeric', 'parsed_reviews'], how='all')

    # Fill cost with median by location and rest_type
    df_clean['cost_numeric'] = df_clean.groupby(['location', 'rest_type'])['cost_numeric'].transform(
        lambda x: x.fillna(x.median())
    )
    df_clean['cost_numeric'] = df_clean['cost_numeric'].fillna(df_clean['cost_numeric'].median())

    # 11. Create restaurant_id
    df_clean['restaurant_id'] = range(1, len(df_clean) + 1)

    # 12. Generate synthetic timestamps for churn analysis
    # In real data, these would come from review timestamps
    df_clean['last_review_date'] = df_clean['review_count'].apply(
        lambda x: datetime.now() - timedelta(days=random.randint(1, 90)) if x > 0 else datetime.now() - timedelta(days=random.randint(90, 365))
    )
    df_clean['days_since_last_review'] = (datetime.now() - df_clean['last_review_date']).dt.days

    print(f"Final cleaned shape: {df_clean.shape}")

    return df_clean


if __name__ == "__main__":
    import random
    random.seed(42)

    # Load raw data
    df_raw = pd.read_csv('../data/zomato.csv')

    # Clean
    df_clean = clean_and_transform(df_raw)

    # Save
    df_clean.to_csv('../data/cleaned_zomato_bangalore.csv', index=False)

    # Also save exploded reviews for NLP analysis
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
    df_reviews.to_csv('../data/exploded_reviews.csv', index=False)

    print(f"\nSaved cleaned data and {len(df_reviews)} individual reviews.")
