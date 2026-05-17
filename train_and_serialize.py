import os
import pandas as pd
import pickle
from src.churn_prediction import ChurnPredictor

print("Loading cleaned data...")
df_clean = pd.read_csv('data/cleaned_zomato_bangalore.csv')
print("Loading sentiment summary...")
summary = pd.read_csv('data/restaurant_sentiment_summary.csv')

print("Engineering churn features...")
predictor = ChurnPredictor()
df_features = predictor.engineer_churn_features(df_clean, summary)

print("Preparing ML features...")
X, y, feature_names = predictor.prepare_ml_features(df_features)

print("Training models...")
results, best_model_name = predictor.train_and_evaluate(X, y, feature_names)

print("Predicting risk scoring...")
churn_probs, risk_cats = predictor.predict_risk(X)
df_features['churn_probability'] = churn_probs
df_features['risk_category'] = risk_cats

print("Plotting model evaluation...")
predictor.plot_results(results, best_model_name, 'reports/churn_model_evaluation.png')

print("Dropping heavy text columns to save space...")
heavy_cols = ['reviews_list', 'parsed_reviews', 'all_review_texts', 'menu_item', 'dish_liked', 'url', 'address', 'phone']
df_features_light = df_features.drop(columns=[col for col in heavy_cols if col in df_features.columns])
df_features_light.to_csv('data/restaurants_with_churn_features.csv', index=False)
print("Saved light features with churn risk predictions to data/restaurants_with_churn_features.csv")

print("Serializing best model and scaler...")
os.makedirs('models', exist_ok=True)
with open('models/churn_model.pkl', 'wb') as f:
    pickle.dump(predictor.best_model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(predictor.scaler, f)
print("Successfully serialized churn model and scaler!")
