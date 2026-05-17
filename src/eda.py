"""
Exploratory Data Analysis Module
Cloud Kitchen Sentiment & Churn Analysis

Generates:
- Distribution plots for ratings, costs, votes
- Location-based analysis
- Cuisine popularity analysis
- Correlation heatmaps
- Review volume trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_rating_distribution(df, save_path=None):
    """Distribution of restaurant ratings."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    sns.histplot(df['rate_numeric'].dropna(), bins=20, kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title('Distribution of Restaurant Ratings', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Rating (out of 5)')
    axes[0].set_ylabel('Count')
    axes[0].axvline(df['rate_numeric'].mean(), color='red', linestyle='--', label=f'Mean: {df["rate_numeric"].mean():.2f}')
    axes[0].legend()

    # Box plot by online order
    sns.boxplot(data=df, x='online_order', y='rate_numeric', ax=axes[1], palette='Set2')
    axes[1].set_title('Rating Distribution by Online Order Availability')
    axes[1].set_xlabel('Accepts Online Orders')
    axes[1].set_ylabel('Rating')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cost_analysis(df, save_path=None):
    """Cost analysis by location and restaurant type."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Cost distribution
    sns.histplot(df['cost_numeric'].dropna(), bins=30, kde=True, ax=axes[0,0], color='coral')
    axes[0,0].set_title('Distribution of Cost for Two People')
    axes[0,0].set_xlabel('Cost (INR)')

    # Top 10 expensive localities
    loc_cost = df.groupby('location')['cost_numeric'].mean().sort_values(ascending=False).head(10)
    loc_cost.plot(kind='barh', ax=axes[0,1], color='teal')
    axes[0,1].set_title('Top 10 Most Expensive Localities (Avg Cost)')
    axes[0,1].set_xlabel('Average Cost (INR)')

    # Cost vs Rating scatter
    sample_df = df.sample(min(1000, len(df)))
    sns.scatterplot(data=sample_df, x='rate_numeric', y='cost_numeric', 
                    hue='online_order', alpha=0.6, ax=axes[1,0])
    axes[1,0].set_title('Cost vs Rating Correlation')
    axes[1,0].set_xlabel('Rating')
    axes[1,0].set_ylabel('Cost (INR)')

    # Cost by restaurant type
    rest_cost = df.groupby('rest_type')['cost_numeric'].mean().sort_values(ascending=False).head(10)
    rest_cost.plot(kind='bar', ax=axes[1,1], color='purple', rot=45)
    axes[1,1].set_title('Average Cost by Restaurant Type')
    axes[1,1].set_ylabel('Average Cost (INR)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_location_analysis(df, save_path=None):
    """Location-based restaurant analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top locations by restaurant count
    loc_counts = df['location'].value_counts().head(10)
    loc_counts.plot(kind='bar', ax=axes[0,0], color='navy')
    axes[0,0].set_title('Top 10 Locations by Restaurant Count')
    axes[0,0].set_ylabel('Number of Restaurants')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Average rating by location
    loc_rating = df.groupby('location')['rate_numeric'].mean().sort_values(ascending=False).head(10)
    loc_rating.plot(kind='barh', ax=axes[0,1], color='green')
    axes[0,1].set_title('Top 10 Locations by Average Rating')
    axes[0,1].set_xlabel('Average Rating')

    # Votes by location
    loc_votes = df.groupby('location')['votes_numeric'].sum().sort_values(ascending=False).head(10)
    loc_votes.plot(kind='bar', ax=axes[1,0], color='orange')
    axes[1,0].set_title('Top 10 Locations by Total Votes')
    axes[1,0].set_ylabel('Total Votes')
    axes[1,0].tick_params(axis='x', rotation=45)

    # Restaurant type distribution in top location
    top_loc = loc_counts.index[0]
    type_dist = df[df['location'] == top_loc]['rest_type'].value_counts().head(8)
    type_dist.plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%', startangle=90)
    axes[1,1].set_title(f'Restaurant Types in {top_loc}')
    axes[1,1].set_ylabel('')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cuisine_analysis(df, save_path=None):
    """Cuisine popularity and performance analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Extract all cuisines
    all_cuisines = []
    for cuisines in df['cuisines'].dropna():
        all_cuisines.extend([c.strip() for c in str(cuisines).split(',')])

    cuisine_counts = Counter(all_cuisines)
    top_cuisines = dict(cuisine_counts.most_common(10))

    pd.Series(top_cuisines).plot(kind='bar', ax=axes[0,0], color='crimson')
    axes[0,0].set_title('Top 10 Most Popular Cuisines')
    axes[0,0].set_ylabel('Number of Restaurants')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Cuisine vs rating
    cuisine_ratings = {}
    for cuisine in list(top_cuisines.keys())[:8]:
        mask = df['cuisines'].str.contains(cuisine, na=False)
        cuisine_ratings[cuisine] = df[mask]['rate_numeric'].mean()

    pd.Series(cuisine_ratings).sort_values(ascending=False).plot(kind='barh', ax=axes[0,1], color='darkgreen')
    axes[0,1].set_title('Average Rating by Cuisine')
    axes[0,1].set_xlabel('Average Rating')

    # Online order vs cuisine
    online_cuisine = {}
    for cuisine in list(top_cuisines.keys())[:6]:
        mask = df['cuisines'].str.contains(cuisine, na=False)
        online_pct = df[mask]['online_order_binary'].mean() * 100
        online_cuisine[cuisine] = online_pct

    pd.Series(online_cuisine).plot(kind='bar', ax=axes[1,0], color='skyblue')
    axes[1,0].set_title('Online Order Adoption by Cuisine (%)')
    axes[1,0].set_ylabel('% Accepting Online Orders')
    axes[1,0].tick_params(axis='x', rotation=45)

    # Correlation heatmap
    numeric_cols = ['rate_numeric', 'cost_numeric', 'votes_numeric', 'review_count', 'online_order_binary', 'book_table_binary']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1], fmt='.2f')
    axes[1,1].set_title('Feature Correlation Matrix')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_review_volume_analysis(df_reviews, save_path=None):
    """Analyze review volume patterns for churn indicators."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Review count distribution
    sns.histplot(df_reviews.groupby('restaurant_id').size(), bins=30, ax=axes[0], color='mediumpurple')
    axes[0].set_title('Distribution of Reviews per Restaurant')
    axes[0].set_xlabel('Number of Reviews')
    axes[0].set_ylabel('Number of Restaurants')

    # Review rating distribution
    sns.histplot(df_reviews['review_rating'].dropna(), bins=10, ax=axes[1], color='gold')
    axes[1].set_title('Distribution of Individual Review Ratings')
    axes[1].set_xlabel('Review Rating')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv('../data/cleaned_zomato_bangalore.csv')
    df_reviews = pd.read_csv('../data/exploded_reviews.csv')

    print("Generating EDA visualizations...")

    plot_rating_distribution(df, '../reports/rating_distribution.png')
    plot_cost_analysis(df, '../reports/cost_analysis.png')
    plot_location_analysis(df, '../reports/location_analysis.png')
    plot_cuisine_analysis(df, '../reports/cuisine_analysis.png')
    plot_review_volume_analysis(df_reviews, '../reports/review_volume_analysis.png')

    print("EDA complete. Reports saved to ../reports/")
