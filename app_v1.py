import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="IMDb Rating Predictor",
    layout="wide"
)

st.title("IMDb Movie Rating Predictor")
st.write("Enter the movie details below to predict its IMDb rating category.")

# ==========================================================
# Load Saved Objects
# ==========================================================

@st.cache_resource
def load_artifacts():
    return {
        "model": joblib.load("models/best_model.pkl"),
        "label_encoder": joblib.load("models/label_encoder.pkl"),
        "scaler": joblib.load("models/scaler.pkl"),
        "cv_genres": joblib.load("models/cv_genres.pkl"),
        "tfidf_kw": joblib.load("models/tfidf_kw.pkl"),
        "selected_features": joblib.load("models/selected_features.pkl"),
        "freq_lookups": joblib.load("models/freq_lookups.pkl"),
        "rare_category_maps": joblib.load("models/rare_category_maps.pkl"),
        "num_cols": joblib.load("models/num_cols.pkl"),
        "category_values": joblib.load("models/category_values.pkl"),
    }

artifacts = load_artifacts()

model = artifacts["model"]
label_encoder = artifacts["label_encoder"]
scaler = artifacts["scaler"]
cv_genres = artifacts["cv_genres"]
tfidf_kw = artifacts["tfidf_kw"]
selected_features = artifacts["selected_features"]
freq_lookups = artifacts["freq_lookups"]
rare_category_maps = artifacts["rare_category_maps"]
num_cols = artifacts["num_cols"]
category_values = artifacts["category_values"]

# ==========================================================
# User Inputs Form Structure
# ==========================================================

user_input = {}

tab1, tab2, tab3 = st.tabs(["General & Overview", "Cast & Crew", "Metrics & Engagement"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        user_input['title_year'] = st.number_input("Release Year", min_value=1900, max_value=2026, value=2020)
        user_input['duration'] = st.number_input("Duration (minutes)", min_value=1, max_value=500, value=120)
        user_input['color'] = st.selectbox("Color / B&W", options=["Color", "Black and White"])
        user_input['content_rating'] = st.selectbox("Content Rating", options=category_values['content_rating'])
    with col2:
        user_input['language'] = st.selectbox("Language", options=category_values['language'])
        user_input['country'] = st.selectbox("Country", options=category_values['country'])
        user_input['genres'] = st.multiselect("Genres", options=category_values['genres'], default=["Action"])
        user_input['plot_keywords'] = st.text_input("Plot Keywords (space-separated)", value="hero battle future")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        user_input['director_name'] = st.selectbox("Director Name", options=category_values['director_name'])
        user_input['director_facebook_likes'] = st.number_input("Director Facebook Likes", min_value=0, value=500)
        user_input['actor_1_name'] = st.selectbox("Lead Actor Name", options=category_values['actor_1_name'])
        user_input['actor_1_facebook_likes'] = st.number_input("Lead Actor Facebook Likes", min_value=0, value=1000)
    with col2:
        user_input['actor_2_name'] = st.selectbox("Supporting Actor 1 Name", options=category_values['actor_2_name'])
        user_input['actor_2_facebook_likes'] = st.number_input("Supporting Actor 1 Facebook Likes", min_value=0, value=500)
        user_input['actor_3_name'] = st.selectbox("Supporting Actor 2 Name", options=category_values['actor_3_name'])
        user_input['actor_3_facebook_likes'] = st.number_input("Supporting Actor 2 Facebook Likes", min_value=0, value=250)
        user_input['cast_total_facebook_likes'] = st.number_input("Total Cast Facebook Likes", min_value=0, value=2500)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        user_input['budget'] = st.number_input("Budget ($)", min_value=0, value=50000000)
        user_input['gross'] = st.number_input("Gross Earnings ($)", min_value=0, value=100000000)
        user_input['movie_facebook_likes'] = st.number_input("Movie Facebook Likes", min_value=0, value=5000)
        user_input['facenumber_in_poster'] = st.number_input("Faces in Poster", min_value=0, value=1)
    with col2:
        user_input['aspect_ratio'] = st.number_input("Aspect Ratio", min_value=0.0, value=2.35)
        user_input['num_critic_for_reviews'] = st.number_input("Critic Reviews Count", min_value=0, value=150)
        user_input['num_user_for_reviews'] = st.number_input("User Reviews Count", min_value=0, value=300)
        user_input['num_voted_users'] = st.number_input("Total Voted Users", min_value=0, value=50000)

# ==========================================================
# Predict Button & Processing
# ==========================================================

st.write("---")

if st.button("Predict IMDb Rating Category", use_container_width=True):

    input_df = pd.DataFrame([user_input])

    # 1. Feature Engineering & Derived Features
    input_df['log_budget'] = np.log1p(max(input_df['budget'].iloc[0], 0))
    input_df['log_gross'] = np.log1p(max(input_df['gross'].iloc[0], 0))
    input_df['roi'] = (input_df['gross'] - input_df['budget']) / (input_df['budget'] + 1000)
    input_df['review_ratio'] = input_df['num_user_for_reviews'] / (input_df['num_critic_for_reviews'] + 1)
    input_df['votes_per_review'] = input_df['num_voted_users'] / (input_df['num_user_for_reviews'] + 1)
    input_df['actor_lead_share'] = input_df['actor_1_facebook_likes'] / (input_df['cast_total_facebook_likes'] + 1)
    input_df['director_share'] = input_df['director_facebook_likes'] / (input_df['cast_total_facebook_likes'] + 1)
    input_df['movie_age'] = 2026 - input_df['title_year'].iloc[0]
    input_df['is_color'] = 1 if input_df['color'].iloc[0] == 'Color' else 0

    # 2. Text Vectorization (Genres & Keywords)
    genres_str = " ".join(input_df['genres'].iloc[0]) if isinstance(input_df['genres'].iloc[0], list) else str(input_df['genres'].iloc[0])
    genres_encoded = pd.DataFrame(
        cv_genres.transform([genres_str]).toarray(),
        columns=[f"genre_{w}" for w in cv_genres.get_feature_names_out()]
    )

    keywords_str = str(input_df['plot_keywords'].iloc[0]).replace('|', ' ')
    keywords_encoded = pd.DataFrame(
        tfidf_kw.transform([keywords_str]).toarray(),
        columns=[f"kw_{w}" for w in tfidf_kw.get_feature_names_out()]
    )

    # 3. Frequency Encoding for High-Cardinality Names
    freq_data = {}
    for col in ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']:
        val = input_df[col].iloc[0]
        freq_data[f'{col}_freq'] = freq_lookups[col].get(val, 0)
    freq_df = pd.DataFrame([freq_data])

    # 4. Rare Grouping & One-Hot Flags for Categoricals
    cat_flags = {}
    for col in ['country', 'language', 'content_rating']:
        val = input_df[col].iloc[0]
        grouped_val = val if val in rare_category_maps[col] else 'Other'
        cat_flags[f'{col}_grouped_{grouped_val}'] = 1
    cat_df = pd.DataFrame([cat_flags])

    # 5. Scale Numerical Variables
    scaled_num = input_df[num_cols].copy()
    scaled_num[num_cols] = scaler.transform(scaled_num[num_cols])

    # 6. Combine Features & Align to Selected Model Features
    processed = pd.concat([scaled_num, freq_df, genres_encoded, keywords_encoded, cat_df], axis=1)
    processed = processed.reindex(columns=selected_features, fill_value=0)

    # 7. Model Prediction
    prediction_idx = model.predict(processed)[0]
    prediction = label_encoder.inverse_transform([prediction_idx])[0]

    # 8. Display Results
    st.success(f"Predicted IMDb Rating Category: **{prediction}**")