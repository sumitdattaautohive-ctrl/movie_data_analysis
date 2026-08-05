# ==========================================
# Cell 1 - Imports
# ==========================================

# Data Manipulation
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing
from sklearn.impute import SimpleImputer

# Warnings
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Pandas Display Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 100)

# Matplotlib & Seaborn Settings
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

print("All libraries imported successfully.")


# ==========================================
# Cell 2 - Configuration
# ==========================================

# File Paths
INPUT_FILE = "movie_metadata.csv"
OUTPUT_FILE = "cleaned_movie_data.csv"

# Target Variable
TARGET_COLUMN = "imdb_score"
TARGET_LABEL = "imdb_binned"

# Target Binning
IMDB_BINS = [1, 3, 6, 10]
IMDB_LABELS = ["FLOP", "AVG", "HIT"]

# Columns to Remove
DROP_COLUMNS = [
    "movie_title",
    "movie_imdb_link"
]

# Missing Value Strategies
NUMERICAL_IMPUTATION = "median"
CATEGORICAL_IMPUTATION = "most_frequent"

print("Configuration Loaded Successfully")


# ==========================================
# Cell 3 - Utility Functions
# ==========================================

def print_header(title):
    """
    Print a formatted section header.
    """
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)


def load_dataset(file_path):
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(file_path)

    print(f"Dataset loaded successfully from '{file_path}'")
    print(f"Shape : {df.shape}")

    return df


def get_column_types(df):
    """
    Return numerical and categorical columns.
    """
    numerical_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = df.select_dtypes(include="object").columns.tolist()

    return numerical_columns, categorical_columns


def dataset_summary(df):
    """
    Display basic dataset information.
    """
    print_header("Dataset Summary")

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nData Types")
    print(df.dtypes)

    print("\nFirst 5 Records")
    print(df.head())


def missing_value_summary(df):
    """
    Return missing value summary.
    """
    summary = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Missing Percentage (%)":
            (df.isnull().mean() * 100).round(2)
    })

    return summary.sort_values(
        by="Missing Values",
        ascending=False
    )


def duplicate_summary(df):
    """
    Return duplicate statistics.
    """
    duplicate_count = df.duplicated().sum()

    duplicate_percentage = (
                                   duplicate_count / len(df)
                           ) * 100

    return duplicate_count, duplicate_percentage


def print_separator():
    """
    Print separator line.
    """
    print("-" * 60)


# ==========================================
# Cell 4 - Load Dataset
# ==========================================

print_header("Load Dataset")

# Load dataset
df = load_dataset(INPUT_FILE)

# Create a backup copy
df_original = df.copy()

print("\nDataset loaded successfully.")
print(f"Shape : {df.shape}")

print_separator()

print("First 5 Records")
print(df.head())

