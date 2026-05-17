"""
NLP Sentiment Analysis Pipeline
Cloud Kitchen Sentiment & Churn Analysis

Techniques:
- VADER sentiment scoring (optimized for social media/short text)
- TextBlob polarity/subjectivity
- Custom operational complaint classifier
- Aspect-based sentiment extraction
- Topic modeling for review categorization
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import NLP libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VADER not available. Using fallback sentiment method.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


class SentimentAnalyzer:
    """Comprehensive sentiment analysis for restaurant reviews."""

    def __init__(self):
        self.analyzer = None
        if VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()

        # Custom lexicon for food delivery domain
        self.operational_keywords = {
            'delivery_delay': ['late', 'delay', 'waited', 'hours', 'slow', 'forever', 'never arrived'],
            'food_quality': ['cold', 'stale', 'spoiled', 'undercooked', 'burnt', 'soggy', 'bland'],
            'packaging': ['spilled', 'leaked', 'broken', 'packaging', 'container', 'messy'],
            'service': ['rude', 'unresponsive', 'unprofessional', 'behavior', 'attitude'],
            'hygiene': ['hair', 'dirty', 'hygiene', 'unclean', 'rotten', 'smelly'],
            'wrong_order': ['wrong', 'missing', 'cancelled', 'refund', 'incorrect']
        }

        self.positive_food_words = ['delicious', 'tasty', 'amazing', 'excellent', 'perfect', 
                                   'fresh', 'flavorful', 'authentic', 'crispy', 'soft', 'hot',
                                   'yummy', 'mouthwatering', 'heavenly', 'awesome', 'great',
                                   'best', 'love', 'loved', 'fantastic', 'superb', 'good']

        self.negative_food_words = ['bad', 'terrible', 'awful', 'worst', 'horrible', 'disgusting',
                                   'bland', 'oily', 'salty', 'sweet', 'sour', 'bitter',
                                   'overcooked', 'undercooked', 'chewy', 'hard', 'dry']

    def vader_sentiment(self, text):
        """Get VADER compound sentiment score."""
        if not self.analyzer or pd.isna(text):
            return 0.0
        scores = self.analyzer.polarity_scores(str(text))
        return scores['compound']

    def textblob_sentiment(self, text):
        """Get TextBlob polarity and subjectivity."""
        if not TEXTBLOB_AVAILABLE or pd.isna(text):
            return 0.0, 0.0
        blob = TextBlob(str(text))
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    def fallback_sentiment(self, text):
        """Simple lexicon-based sentiment if libraries unavailable."""
        if pd.isna(text):
            return 0.0

        text = str(text).lower()
        pos_count = sum(1 for word in self.positive_food_words if word in text)
        neg_count = sum(1 for word in self.negative_food_words if word in text)

        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    def classify_sentiment(self, score, method='vader'):
        """Classify sentiment into categories."""
        if method == 'vader':
            if score >= 0.05:
                return 'Positive'
            elif score <= -0.05:
                return 'Negative'
            else:
                return 'Neutral'
        else:
            if score > 0.1:
                return 'Positive'
            elif score < -0.1:
                return 'Negative'
            else:
                return 'Neutral'

    def detect_operational_issues(self, text):
        """Detect specific operational complaint categories."""
        if pd.isna(text):
            return {}

        text = str(text).lower()
        issues = {}

        for category, keywords in self.operational_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            issues[category] = score > 0

        issues['any_operational_issue'] = any(issues.values())
        return issues

    def extract_aspects(self, text):
        """Extract food aspects mentioned in review."""
        if pd.isna(text):
            return []

        text = str(text).lower()
        aspects = []

        aspect_keywords = {
            'taste': ['taste', 'flavor', 'delicious', 'bland', 'spicy', 'sweet'],
            'portion': ['portion', 'quantity', 'size', 'small', 'large', 'enough'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'worth', 'value', 'overpriced'],
            'service': ['service', 'staff', 'waiter', 'delivery', 'rider'],
            'ambiance': ['ambiance', 'atmosphere', 'decor', 'clean', 'hygiene'],
            'packaging': ['packaging', 'box', 'container', 'spill', 'leak']
        }

        for aspect, keywords in aspect_keywords.items():
            if any(kw in text for kw in keywords):
                aspects.append(aspect)

        return aspects

    def analyze_reviews(self, df_reviews):
        """
        Main analysis pipeline for review DataFrame.

        Args:
            df_reviews: DataFrame with 'review_text' column

        Returns:
            DataFrame with sentiment scores and operational flags
        """
        df = df_reviews.copy()

        print("Running sentiment analysis on reviews...")

        # VADER sentiment
        if VADER_AVAILABLE:
            df['vader_compound'] = df['review_text'].apply(self.vader_sentiment)
            df['sentiment_label'] = df['vader_compound'].apply(lambda x: self.classify_sentiment(x, 'vader'))
        else:
            df['vader_compound'] = df['review_text'].apply(self.fallback_sentiment)
            df['sentiment_label'] = df['vader_compound'].apply(lambda x: self.classify_sentiment(x, 'fallback'))

        # TextBlob (if available)
        if TEXTBLOB_AVAILABLE:
            tb_results = df['review_text'].apply(self.textblob_sentiment)
            df['textblob_polarity'] = [r[0] for r in tb_results]
            df['textblob_subjectivity'] = [r[1] for r in tb_results]

        # Operational issue detection
        print("Detecting operational issues...")
        op_results = df['review_text'].apply(self.detect_operational_issues)
        op_df = pd.DataFrame(list(op_results))
        df = pd.concat([df, op_df], axis=1)

        # Aspect extraction
        print("Extracting aspects...")
        df['aspects'] = df['review_text'].apply(self.extract_aspects)
        df['aspect_count'] = df['aspects'].apply(len)

        # Sentiment-review rating agreement
        df['sentiment_rating_agreement'] = df.apply(
            lambda row: 'Agree' if (
                (row['sentiment_label'] == 'Positive' and row['review_rating'] >= 4) or
                (row['sentiment_label'] == 'Negative' and row['review_rating'] <= 2) or
                (row['sentiment_label'] == 'Neutral' and 2.5 <= row['review_rating'] <= 3.5)
            ) else 'Disagree',
            axis=1
        )

        return df

    def generate_restaurant_sentiment_summary(self, df_analyzed):
        """Aggregate sentiment metrics per restaurant."""
        summary = df_analyzed.groupby('restaurant_id').agg({
            'vader_compound': ['mean', 'std', 'min', 'max'],
            'sentiment_label': lambda x: (x == 'Negative').sum() / len(x) * 100,
            'any_operational_issue': 'mean',
            'delivery_delay': 'sum',
            'food_quality': 'sum',
            'packaging': 'sum',
            'service': 'sum',
            'hygiene': 'sum',
            'wrong_order': 'sum',
            'aspect_count': 'mean',
            'sentiment_rating_agreement': lambda x: (x == 'Agree').sum() / len(x) * 100
        }).reset_index()

        # Flatten column names
        summary.columns = [
            'restaurant_id', 'avg_sentiment', 'sentiment_std', 'min_sentiment', 'max_sentiment',
            'negative_review_pct', 'operational_issue_rate', 'delivery_delay_count',
            'food_quality_count', 'packaging_count', 'service_count', 'hygiene_count',
            'wrong_order_count', 'avg_aspect_count', 'sentiment_rating_agreement_pct'
        ]

        return summary


if __name__ == "__main__":
    # Load data
    df_reviews = pd.read_csv('../data/exploded_reviews.csv')

    # Analyze
    analyzer = SentimentAnalyzer()
    df_analyzed = analyzer.analyze_reviews(df_reviews)

    # Save analyzed reviews
    df_analyzed.to_csv('../data/reviews_with_sentiment.csv', index=False)

    # Generate restaurant-level summary
    summary = analyzer.generate_restaurant_sentiment_summary(df_analyzed)
    summary.to_csv('../data/restaurant_sentiment_summary.csv', index=False)

    print(f"\nAnalyzed {len(df_analyzed)} reviews")
    print(f"Sentiment distribution:")
    print(df_analyzed['sentiment_label'].value_counts())
    print(f"\nOperational issues detected:")
    print(f"Delivery delays: {df_analyzed['delivery_delay'].sum()}")
    print(f"Food quality issues: {df_analyzed['food_quality'].sum()}")
    print(f"Packaging issues: {df_analyzed['packaging'].sum()}")
    print(f"Service issues: {df_analyzed['service'].sum()}")
    print(f"Hygiene issues: {df_analyzed['hygiene'].sum()}")
    print(f"Wrong orders: {df_analyzed['wrong_order'].sum()}")
