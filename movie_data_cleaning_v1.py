# ==========================================
# Section 1 - Imports & Settings
# ==========================================
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Display & Warning Settings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 100)

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

print("All libraries imported successfully.")

# ==========================================
# Section 2 - Configuration
# ==========================================
INPUT_FILE = "movie_metadata.csv"
OUTPUT_FILE = "cleaned_movie_data_v2.csv"

# Target Column Setup
TARGET_COLUMN = "imdb_score"
TARGET_LABEL = "imdb_binned"
IMDB_BINS = [1, 3, 6, 10]
IMDB_LABELS = ["FLOP", "AVG", "HIT"]

# Unnecessary Columns to Remove
DROP_COLUMNS = ["movie_title", "movie_imdb_link"]

# Imputation Strategies for Remaining General Columns
NUMERICAL_IMPUTATION = "median"
CATEGORICAL_IMPUTATION = "most_frequent"

print("Configuration Loaded Successfully.")

# ==========================================
# Section 3 - Data Cleaning & Processing
# ==========================================

# 1. Load Dataset
print("\n" + "=" * 60)
print("1. LOAD DATASET")
print("=" * 60)
df = pd.read_csv(INPUT_FILE)
df_original = df.copy()

print(f"Dataset loaded successfully from '{INPUT_FILE}'")
print(f"Initial Shape: {df.shape}")

# 2. Remove Duplicates
print("\n" + "=" * 60)
print("2. REMOVE DUPLICATES")
print("=" * 60)
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {duplicate_count} duplicate row(s).")
else:
    print("No duplicate rows found.")
print(f"Current Shape: {df.shape}")

# 3. Drop Unnecessary Columns
print("\n" + "=" * 60)
print("3. DROP UNNECESSARY COLUMNS")
print("=" * 60)
cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped columns: {cols_to_drop}")
else:
    print("No specified columns found to drop.")
print(f"Current Shape: {df.shape}")

# 4. Specific Column Imputation & Feature Engineering
print("\n" + "=" * 60)
print("4. SPECIFIC IMPUTATION & FEATURE ENGINEERING")
print("=" * 60)

# Fill missing values
df['budget'] = df['budget'].fillna(df['budget'].median())
df['gross'] = df['gross'].fillna(df['gross'].median())
df['director_facebook_likes'] = df['director_facebook_likes'].fillna(0)
df['actor_1_facebook_likes'] = df['actor_1_facebook_likes'].fillna(0)
df['actor_2_facebook_likes'] = df['actor_2_facebook_likes'].fillna(0)
df['actor_3_facebook_likes'] = df['actor_3_facebook_likes'].fillna(0)
df['cast_total_facebook_likes'] = df['cast_total_facebook_likes'].fillna(0)
df['num_critic_for_reviews'] = df['num_critic_for_reviews'].fillna(0)
df['num_user_for_reviews'] = df['num_user_for_reviews'].fillna(0)
df['num_voted_users'] = df['num_voted_users'].fillna(0)
df['title_year'] = df['title_year'].fillna(df['title_year'].median())

# Binary Feature Creation
df['is_color'] = (df['color'].fillna('') == 'Color').astype(int)

# Financial & Interaction Features
df['log_budget'] = np.log1p(np.maximum(df['budget'], 0))
df['log_gross'] = np.log1p(np.maximum(df['gross'], 0))
df['roi'] = (df['gross'] - df['budget']) / (df['budget'] + 1000)
df['review_ratio'] = df['num_user_for_reviews'] / (df['num_critic_for_reviews'] + 1)
df['votes_per_review'] = df['num_voted_users'] / (df['num_user_for_reviews'] + 1)
df['actor_lead_share'] = df['actor_1_facebook_likes'] / (df['cast_total_facebook_likes'] + 1)
df['director_share'] = df['director_facebook_likes'] / (df['cast_total_facebook_likes'] + 1)
df['movie_age'] = 2026 - df['title_year']

print("Domain-specific missing values handled and engineered features generated.")

# 5. Handle Remaining Missing Values
print("\n" + "=" * 60)
print("5. HANDLE REMAINING MISSING VALUES")
print("=" * 60)

num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# Exclude target score from generic imputation
if TARGET_COLUMN in num_cols:
    num_cols.remove(TARGET_COLUMN)

# Impute Remaining Numerical Columns
if num_cols:
    num_imputer = SimpleImputer(strategy=NUMERICAL_IMPUTATION)
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    print(f"Imputed remaining numerical columns using strategy='{NUMERICAL_IMPUTATION}'")

# Impute Remaining Categorical Columns
if cat_cols:
    cat_imputer = SimpleImputer(strategy=CATEGORICAL_IMPUTATION)
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    print(f"Imputed remaining categorical columns using strategy='{CATEGORICAL_IMPUTATION}'")

print(f"Remaining missing values across dataset: {df.isnull().sum().sum()}")

# 6. Bin Target Variable
print("\n" + "=" * 60)
print("6. CREATE TARGET BINS")
print("=" * 60)

if TARGET_COLUMN in df.columns:
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df[TARGET_LABEL] = pd.cut(
        df[TARGET_COLUMN],
        bins=IMDB_BINS,
        labels=IMDB_LABELS,
        include_lowest=True
    )

    df.drop(columns=[TARGET_COLUMN], inplace=True)
    print(f"Transformed '{TARGET_COLUMN}' into '{TARGET_LABEL}'")
    print("\nTarget Class Distribution:")
    print(df[TARGET_LABEL].value_counts())

# 7. Save Cleaned Dataset
print("\n" + "=" * 60)
print("7. SAVE CLEANED DATASET")
print("=" * 60)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Cleaned dataset saved successfully to '{OUTPUT_FILE}'")
print(f"Final Cleaned Shape: {df.shape}")