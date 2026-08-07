# Imports
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
print("All Imports Are Successful")

# Load Cleaned Data
data = pd.read_csv("cleaned_movie_data_v2.csv")
df = data.copy()

# 1. Handle Delimited Text (Genres & Plot Keywords Vectorization)
genres_clean = df['genres'].fillna('').apply(lambda x: str(x).replace('|', ' '))
cv_genres = CountVectorizer(min_df=5)
genres_encoded = pd.DataFrame(
    cv_genres.fit_transform(genres_clean).toarray(),
    columns=[f"genre_{w}" for w in cv_genres.get_feature_names_out()],
    index=df.index
)

keywords_clean = df['plot_keywords'].fillna('').apply(lambda x: str(x).replace('|', ' '))
tfidf_kw = TfidfVectorizer(max_features=50, min_df=3)
keywords_encoded = pd.DataFrame(
    tfidf_kw.fit_transform(keywords_clean).toarray(),
    columns=[f"kw_{w}" for w in tfidf_kw.get_feature_names_out()],
    index=df.index
)

# 2. Frequency Encoding for High-Cardinality Names
high_card_cols = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']
freq_lookups = {}
freq_dfs = []

for col in high_card_cols:
    counts = df[col].value_counts().to_dict()
    freq_lookups[col] = counts
    freq_dfs.append(pd.Series(df[col].map(counts).fillna(0), name=f'{col}_freq'))

freq_df = pd.concat(freq_dfs, axis=1)

# 3. Rare Grouping for Medium-Cardinality Categoricals
rare_category_maps = {}
for col in ['country', 'language', 'content_rating']:
    counts = df[col].value_counts()
    valid_cats = counts[counts >= 10].index.tolist()
    rare_category_maps[col] = valid_cats
    df[f'{col}_grouped'] = df[col].apply(lambda x: x if x in valid_cats else 'Other')

cat_grouped_encoded = pd.get_dummies(
    df[['country_grouped', 'language_grouped', 'content_rating_grouped']],
    drop_first=True,
    dtype=int
)

# Numerical Base Columns
num_cols = [
    'num_critic_for_reviews', 'duration', 'director_facebook_likes',
    'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',
    'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster',
    'num_user_for_reviews', 'budget', 'title_year', 'actor_2_facebook_likes',
    'aspect_ratio', 'movie_facebook_likes', 'is_color', 'log_budget', 'log_gross',
    'roi', 'review_ratio', 'votes_per_review', 'actor_lead_share', 'director_share',
    'movie_age'
]

# Separate Target and Features
label_encoder = LabelEncoder()
df["imdb_binned"] = label_encoder.fit_transform(df["imdb_binned"])

X_all = pd.concat([df[num_cols], freq_df, genres_encoded, keywords_encoded, cat_grouped_encoded], axis=1)
y = df["imdb_binned"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Feature Scaling (Robust)
scaler = RobustScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# Correlation Heatmap
corr_matrix = X_train_scaled[num_cols].corr().abs()
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Numerical Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Feature Selection
feature_selector = GradientBoostingClassifier(
    n_estimators=150,
    random_state=42
)

feature_selector.fit(X_train_scaled, y_train)

feature_importance = pd.Series(
    feature_selector.feature_importances_,
    index=X_train_scaled.columns
).sort_values(ascending=False)

N_FEATURES = 30
selected_features = feature_importance.head(N_FEATURES).index.tolist()

print("=" * 60)
print("FEATURE SELECTION")
print("=" * 60)
print("Original number of features:", X_train_scaled.shape[1])
print("Selected number of features:", len(selected_features))
print("\nSelected Features:")
for feature in selected_features:
    print(feature)

X_train_selected = X_train_scaled[selected_features].copy()
X_test_selected = X_test_scaled[selected_features].copy()

# Multiple Models Setup
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight='balanced'
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss")
}

# Compute sample weights to balance rare classes
sample_weights_train = compute_sample_weight('balanced', y_train)

# Multiple Model Training & Evaluation
results = []

for name, model in models.items():
    if name in ["Gradient Boosting", "AdaBoost", "XGBoost"]:
        model.fit(X_train_selected, y_train, sample_weight=sample_weights_train)
    else:
        model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="macro", zero_division=0)
    })

    print("=" * 60)
    print(name)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print("\n--- Model Performance Comparison ---")
print(results_df)

# Cross Validation
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train_selected,
        y_train,
        cv=5,
        scoring='f1_macro'
    )
    cv_results[name] = scores

    print("=" * 60)
    print(name)
    print("Cross Validation Macro F1 Scores :", scores)
    print("Mean CV Macro F1 :", scores.mean())
    print("Standard Deviation :", scores.std())

cv_df = pd.DataFrame({
    "Model": cv_results.keys(),
    "CV Macro F1": [np.mean(i) for i in cv_results.values()],
    "CV Std": [np.std(i) for i in cv_results.values()]
}).sort_values("CV Macro F1", ascending=False)

print("\n--- Cross Validation Summary ---")
print(cv_df)

# ==========================================
# Hyperparameter Tuning
# ==========================================
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid.fit(X_train_selected, y_train)

print("\nBest Parameters (Random Forest):", grid.best_params_)
print("Best CV Macro F1 Score:", grid.best_score_)

best_model = grid.best_estimator_
rf_pred = best_model.predict(X_test_selected)

# Model Evaluation
accuracy = accuracy_score(y_test, rf_pred)
print("\nTuned Random Forest Final Accuracy:", accuracy)
print(classification_report(y_test, rf_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
# plt.show()

# Export Artifacts for Streamlit
os.makedirs("models", exist_ok=True)

# Save all trained base models
for name, model in models.items():
    filename = name.replace(" ", "_").lower()
    joblib.dump(model, f"models/{filename}.pkl")

# Save Best Tuned Random Forest
joblib.dump(best_model, "models/best_random_forest.pkl")
joblib.dump(best_model, "models/best_model.pkl")

# Save Preprocessing Encoders, Scalers, and Vectorizers
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(cv_genres, "models/cv_genres.pkl")
joblib.dump(tfidf_kw, "models/tfidf_kw.pkl")

# Save Feature Information & Lookup Dictionaries
joblib.dump(selected_features, "models/selected_features.pkl")
joblib.dump(freq_lookups, "models/freq_lookups.pkl")
joblib.dump(rare_category_maps, "models/rare_category_maps.pkl")
joblib.dump(num_cols, "models/num_cols.pkl")

# Save Streamlit UI Dropdown Values
category_values = {
    'genres': sorted(list(cv_genres.vocabulary_.keys())),
    'country': df['country'].dropna().unique().tolist(),
    'language': df['language'].dropna().unique().tolist(),
    'content_rating': df['content_rating'].dropna().unique().tolist(),
    'director_name': df['director_name'].dropna().unique().tolist(),
    'actor_1_name': df['actor_1_name'].dropna().unique().tolist(),
    'actor_2_name': df['actor_2_name'].dropna().unique().tolist(),
    'actor_3_name': df['actor_3_name'].dropna().unique().tolist()
}
joblib.dump(category_values, "models/category_values.pkl")

print("\nAll models and Streamlit artifacts saved successfully in 'models/' directory.")