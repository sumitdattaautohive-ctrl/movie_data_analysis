# Generated from: Model_Training_Movie_Success.ipynb
# Converted at: 2026-08-07T06:17:40.189Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Encoding , Feature-Engineering , Feature-Scaling, Multiple Model Traning and Hyperparameter Tuning


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from category_encoders import TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import os

import warnings
warnings.filterwarnings("ignore")
print("All Import Are Successful")

# Load Cleaned Data
data = pd.read_csv(r"../cleaned_movie_data.csv")
data.head()

df= data.copy()

# # Data Pre-Processing


# Split data into Numerical and Categorical Columns
num_columns = df.select_dtypes(include =['number']).columns
print(f"All Numerical Columns Name : ",num_columns)
cat_columns = df.select_dtypes(include =['object']).columns
print(f"All Categorical Columns Name : ",cat_columns)

# Check Cardinality
for col in cat_columns:
    print(f"{col} :",df[col].nunique())

# # Train-Test- split & Feature-Scaling (Robust)


# Separate features and target

label_encoder = LabelEncoder()
df["imdb_binned"] = label_encoder.fit_transform(df["imdb_binned"])

X = df.drop(columns=["imdb_binned"])
y = df["imdb_binned"]

# Identify categorical columns

cat_cols = X.select_dtypes( include=["object", "category"]).columns.tolist()

num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

print(f"Categorical columns:",cat_cols)

print(f"\nNumerical columns:",num_cols)


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split( X,  y, test_size=0.2,  random_state=42,  stratify=y)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("Training shape:", y_train.head(10))
# Target Encoding

encoder = TargetEncoder(cols=cat_cols)

X_train_cat = encoder.fit_transform(
    X_train[cat_cols],
    y_train
)

X_test_cat = encoder.transform(
    X_test[cat_cols]
)

# Robust Scaling
scaler = RobustScaler()

X_train_num = pd.DataFrame(
    scaler.fit_transform(X_train[num_cols]),
    columns=num_cols,
    index=X_train.index
)

X_test_num = pd.DataFrame(
    scaler.transform(X_test[num_cols]),
    columns=num_cols,
    index=X_test.index
)

#combine both
X_train_final = pd.concat(
    [X_train_num, X_train_cat],
    axis=1
)

X_test_final = pd.concat(
    [X_test_num, X_test_cat],
    axis=1
)

#Calculate absolute correlation matrix

corr_matrix = X_train_final.corr().abs()
# Correlation heatmap
plt.figure(figsize=(25, 20))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
#plt.show()

# # Feature Selection using RFECV


# Feature Selection using RFECV
base_estimator = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=0
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

rfecv = RFECV(
    estimator=base_estimator,
    step=1,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

rfecv.fit(
    X_train_final,
    y_train
)

selected_features = X_train_final.columns[
    rfecv.support_]

print(selected_features)

X_train_selected = X_train_final[selected_features]
X_test_selected = X_test_final[selected_features]


# # Multiple Model Using


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="mlogloss"
    )
}

# # Multiple model Training


results = []

for name, model in models.items():

    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )
    })

    print("="*60)
    print(name)
    print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print(results_df)

# # Hyperparameter Tuning (Random Forest)


param_grid = {

    'n_estimators':[100,200,300],

    'max_depth':[5,10,20,None],

    'min_samples_split':[2,5,10],

    'min_samples_leaf':[1,2,4]

}

grid = GridSearchCV(

    RandomForestClassifier(random_state=42),

    param_grid,

    cv=5,

    scoring='accuracy',

    n_jobs=-1

)

grid.fit(X_train_final,y_train)

print(grid.best_params_)

print(grid.best_score_)

best_rf = grid.best_estimator_

best_rf.fit(X_train_selected, y_train)

rf_pred = best_rf.predict(X_test_selected)

cm = confusion_matrix(y_test, rf_pred)
print("#######################################")
print(pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
))
# # Cross Validation


cv_results = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train_final,
        y_train,
        cv=5,
        scoring='accuracy'
    )

    cv_results[name] = scores

    print("="*60)
    print(name)
    print("Cross Validation Scores :", scores)
    print("Mean Accuracy :", scores.mean())
    print("Standard Deviation :", scores.std())

# # Comparison Table


cv_df = pd.DataFrame({
    "Model": cv_results.keys(),
    "CV Accuracy":[np.mean(i) for i in cv_results.values()],
    "CV Std":[np.std(i) for i in cv_results.values()]
})

cv_df.sort_values("CV Accuracy",ascending=False)

# # Model Evaluation


# # Accuracy


accuracy = accuracy_score(y_test, rf_pred)
print("Accuracy:", accuracy)

# # Classification Report


print(classification_report(y_test, rf_pred))

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
#plt.show()

# # Precision Recall & F1 Score


print("Precision :",precision_score(y_test,rf_pred,average='weighted'))

print("Recall :",recall_score(y_test,rf_pred,average='weighted'))

print("F1 Score :",f1_score(y_test,rf_pred,average='weighted'))

# Create model folder
os.makedirs("models", exist_ok=True)

# Save all trained models
for name, model in models.items():

    filename = name.replace(" ", "_").lower()

    joblib.dump(
        model,
        f"models/{filename}.pkl"
    )

print("All models saved successfully")

# Save Best Tuned Random Forest
joblib.dump(
    best_rf,
    "models/best_random_forest.pkl"
)

print("Best Random Forest saved")

# Save preprocessing objects(Target Label Encoder)
joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)
joblib.dump(
    encoder,
    "models/encoder.pkl"
)

# scaler
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# Save feature information (Categorical columns & Numerical columns)
joblib.dump(
    cat_cols,
    "models/categorical_columns.pkl"
)
joblib.dump(
    num_cols,
    "models/numerical_columns.pkl"
)

# Selected RFECV features
joblib.dump(
    selected_features.tolist(),
    "models/selected_features.pkl"
)

# Save dropdown values for Streamlit

category_values = {}

for col in cat_cols:
    category_values[col] = df[col].dropna().unique().tolist()


joblib.dump(
    category_values,
    "models/category_values.pkl"
)