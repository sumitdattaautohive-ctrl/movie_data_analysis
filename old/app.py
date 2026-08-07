import joblib
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
        "model": joblib.load("models/model.pkl"),
        "encoder": joblib.load("models/encoder.pkl"),
        "scaler": joblib.load("models/scaler.pkl"),
        "label_encoder": joblib.load("models/label_encoder.pkl"),
        "cat_cols": joblib.load("models/categorical_columns.pkl"),
        "num_cols": joblib.load("models/numerical_columns.pkl"),
        "selected_features": joblib.load("models/selected_features.pkl"),
        "category_values": joblib.load("models/category_values.pkl"),
    }

artifacts = load_artifacts()

model = artifacts["model"]
encoder = artifacts["encoder"]
scaler = artifacts["scaler"]
label_encoder = artifacts["label_encoder"]

cat_cols = artifacts["cat_cols"]
num_cols = artifacts["num_cols"]

selected_features = artifacts["selected_features"]
category_values = artifacts["category_values"]

# ==========================================================
# User Inputs
# ==========================================================

st.header("Movie Information")

user_input = {}

# ---------- Numerical ----------

for col in num_cols:

    user_input[col] = st.number_input(
        label=col.replace("_", " ").title(),
        value=0.0
    )

# ---------- Categorical ----------

for col in cat_cols:

    user_input[col] = st.selectbox(
        label=col.replace("_", " ").title(),
        options=category_values[col]
    )

# ==========================================================
# Predict Button
# ==========================================================

if st.button("Predict IMDb Rating", use_container_width=True):

    input_df = pd.DataFrame([user_input])

    # -----------------------------
    # Encode categorical variables
    # -----------------------------

    encoded_cat = encoder.transform(
        input_df[cat_cols]
    )

    # -----------------------------
    # Scale numerical variables
    # -----------------------------

    scaled_num = pd.DataFrame(
        scaler.transform(input_df[num_cols]),
        columns=num_cols
    )

    # -----------------------------
    # Merge
    # -----------------------------

    processed = pd.concat(
        [scaled_num, encoded_cat],
        axis=1
    )

    # -----------------------------
    # Keep selected features only
    # -----------------------------

    processed = processed[selected_features]

    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(processed)[0]

    prediction = label_encoder.inverse_transform([prediction])[0]

    # -----------------------------
    # Display
    # -----------------------------

    st.success(f"Predicted IMDb Rating Category: **{prediction}**")