# Cell 1 - Imports
import warnings

import category_encoders
import matplotlib
import numpy
import pandas
import seaborn
import sklearn
import streamlit
import xgboost

warnings.filterwarnings("ignore")
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from xgboost import XGBClassifier
from category_encoders import TargetEncoder
print(f"numpy              : {numpy.__version__}")
print(f"pandas             : {pandas.__version__}")
print(f"matplotlib         : {matplotlib.__version__}")
print(f"seaborn            : {seaborn.__version__}")
print(f"scikit-learn       : {sklearn.__version__}")
print(f"xgboost            : {xgboost.__version__}")
print(f"category-encoders  : {category_encoders.__version__}")
print(f"streamlit          : {streamlit.__version__}")

print("Imports Loaded")

#Cell 2 - Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "imdb_binned"

MODEL_DIR = "models"

#Cell 3 - Utility Functions
def load_dataset(path):
    return pd.read_csv(path)


def save_model(model, filename):
    joblib.dump(model, f"{MODEL_DIR}/{filename}")

def evaluate(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "F1": f1_score(y_true, y_pred, average="macro")
    }

#Cell 4 - Load Dataset
df = load_dataset("cleaned_movie_data.csv")

df.head()

#Cell 5 - Split Dataset
X = df.drop(TARGET, axis=1)

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(df[TARGET])

cat_cols = X.select_dtypes(include="object").columns

num_cols = X.select_dtypes(exclude="object").columns

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    random_state=RANDOM_STATE,
    test_size=TEST_SIZE
)
# Cell 6 - Preprocessing
def preprocess(
        X_train,
        X_test,
        y_train,
        cat_cols,
        num_cols
):

    encoder = TargetEncoder(cols=cat_cols)

    scaler = RobustScaler()

    train_cat = encoder.fit_transform(
        X_train[cat_cols],
        y_train
    )

    test_cat = encoder.transform(
        X_test[cat_cols]
    )

    train_num = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]),
        columns=num_cols,
        index=X_train.index
    )

    test_num = pd.DataFrame(
        scaler.transform(X_test[num_cols]),
        columns=num_cols,
        index=X_test.index
    )

    X_train = pd.concat(
        [train_num, train_cat],
        axis=1
    )

    X_test = pd.concat(
        [test_num, test_cat],
        axis=1
    )

    return X_train, X_test, encoder, scaler

X_train, X_test, encoder, scaler = preprocess(
    X_train,
    X_test,
    y_train,
    cat_cols,
    num_cols
)
# Cell 7 - Correlation
plt.figure(figsize=(18,12))

sns.heatmap(
    X_train.corr(),
    cmap="coolwarm"
)

plt.show()

# Cell 8 - Feature Selection
def feature_selection(
        X_train,
        y_train
):

    selector = RFECV(
        estimator=XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="mlogloss"
        ),
        cv=StratifiedKFold(5),
        scoring="f1_weighted",
        min_features_to_select=10
    )

    selector.fit(
        X_train,
        y_train
    )

    return selector

selector = feature_selection(
    X_train,
    y_train
)

X_train = X_train.loc[:, selector.support_]

X_test = X_test.loc[:, selector.support_]

# Cell 9 - Models
models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(),

    "Random Forest":
        RandomForestClassifier(),

    "Gradient Boosting":
        GradientBoostingClassifier(),

    "AdaBoost":
        AdaBoostClassifier(),

    "XGBoost":
        XGBClassifier(eval_metric="mlogloss")
}

# Cell 10 - Training

results = []

best_model = None
best_score = 0

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    score = evaluate(y_test, pred)

    score["Model"] = name

    results.append(score)

    if score["F1"] > best_score:
        best_score = score["F1"]
        best_model = model

    print(name)
    print(classification_report(y_test, pred))

# Cell 11 - Results
results = pd.DataFrame(results)

results.sort_values(
    "F1",
    ascending=False
)

# ==========================================
# Cell 12 - Export Models & Metadata
# ==========================================

import os
import joblib

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Save trained objects
save_model(best_model, "model.pkl")
save_model(encoder, "encoder.pkl")
save_model(scaler, "scaler.pkl")
save_model(selector, "selector.pkl")
save_model(label_encoder, "label_encoder.pkl")

# Save feature information
save_model(cat_cols.tolist(), "categorical_columns.pkl")
save_model(num_cols.tolist(), "numerical_columns.pkl")
save_model(X.columns.tolist(), "feature_columns.pkl")

# Save selected features after RFECV
selected_features = X_train.columns.tolist()
save_model(selected_features, "selected_features.pkl")

# Save unique values for categorical columns
category_values = {
    col: sorted(df[col].dropna().astype(str).unique().tolist())
    for col in cat_cols
}
save_model(category_values, "category_values.pkl")


# Cell 13 - Prediction Function (Streamlit)
# def predict(input_df):
#
#     model = joblib.load("models/model.pkl")
#     encoder = joblib.load("models/encoder.pkl")
#     scaler = joblib.load("models/scaler.pkl")
#     selector = joblib.load("models/selector.pkl")
#     label_encoder = joblib.load("models/label_encoder.pkl")
#
#     # Apply preprocessing
#     # Select features
#     # Predict
#
#     prediction = model.predict(processed_df)
#
#     return label_encoder.inverse_transform(prediction)[0]

