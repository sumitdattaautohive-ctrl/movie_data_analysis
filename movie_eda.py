# Cell 1 - Imports
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

print("Imports Loaded Successfully")

# Cell 2 - Configuration
DATASET = "cleaned_movie_data.csv"

TARGET = "imdb_binned"

FIGURE_WIDTH = 12
FIGURE_HEIGHT = 6

TOP_N = 10

# Cell 3 - Utility Functions

def get_column_types(df):

    numerical = df.select_dtypes(include="number").columns.tolist()

    categorical = df.select_dtypes(include="object").columns.tolist()

    return numerical, categorical


def show_shape(df):

    print(f"Rows : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

# Cell 3 - Load dataset
def load_dataset(path):
    return pd.read_csv(path)

# Cell 4 - Visualization Functions
def plot_bar(series, title, xlabel, ylabel):

    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    ax = series.plot(
        kind="bar"
    )

    for container in ax.containers:
        ax.bar_label(container)

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

def plot_box(df, column):

    plt.figure(figsize=(10,5))

    sns.boxplot(
        data=df,
        x=TARGET,
        y=column
    )

    plt.title(f"{column} vs {TARGET}")

    plt.show()


def plot_scatter(df, column):

    plt.figure(figsize=(8,5))

    sns.scatterplot(
        data=df,
        x=TARGET,
        y=column
    )

    plt.title(f"{column} vs {TARGET}")

    plt.show()

def plot_categorical(df, column):

    top = df[column].value_counts().head(TOP_N).index

    temp = df[df[column].isin(top)]

    ct = pd.crosstab(
        temp[column],
        temp[TARGET]
    )

    ct.plot(
        kind="bar",
        figsize=(12,5)
    )

    plt.title(f"{column} vs {TARGET}")

    plt.tight_layout()

    plt.show()

# Cell 5 - Load Dataset
df = load_dataset(DATASET)

show_shape(df)

df.head()

# Cell 6 - Dataset Overview
df.info()

df.describe()

df.describe(include="object")

# Missing value
missing_values = df.isnull().sum()

# Missing percentage
missing_percentage = (
        (missing_values / len(df)) * 100
).round(2)

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": missing_values.values,
    "Missing (%)": missing_percentage.values
})

# Display only columns with missing values
missing_df = missing_df[missing_df["Missing Values"] > 0]

if missing_df.empty:
    print("No missing values found in the dataset.")
else:
    print(
        missing_df.sort_values(
            by="Missing Values",
            ascending=False
        )
    )

# Count duplicate rows
duplicate_count = df.duplicated().sum()

duplicate_percentage = (
                               duplicate_count / len(df)
                       ) * 100

print(f"Total Records          : {len(df)}")
print(f"Duplicate Records      : {duplicate_count}")
print(f"Duplicate Percentage   : {duplicate_percentage:.2f}%")

# Cell 7 - Target Distribution
target_distribution = df[TARGET].value_counts()

print(target_distribution)

plot_bar(
    target_distribution,
    "Movie Success Categories",
    "IMDb Category",
    "Movies"
)

# Cell 8 - Column Types
num_cols, cat_cols = get_column_types(df)
print(num_cols)
print(cat_cols)

# Cell 9 - Genre Analysis
genre_df = df.copy()

genre_df["genres"] = genre_df["genres"].str.split("|")

genre_df = genre_df.explode("genres")

hit = (

    genre_df[genre_df[TARGET] == "HIT"]

    .groupby("genres")

    .size()

    .sort_values(ascending=False)

    .head(TOP_N)

)

plot_bar(
    hit,
    "Top Genres for HIT Movies",
    "Genre",
    "Movies"
)

flop = (

    genre_df[genre_df[TARGET] == "FLOP"]

    .groupby("genres")

    .size()

    .sort_values(ascending=False)

    .head(TOP_N)

)

plot_bar(
    flop,
    "Top Genres for FLOP Movies",
    "Genre",
    "Movies"
)

# Cell 10 - Numerical Data Analysis
for column in num_cols:

    plot_box(df, column)

for column in num_cols:

    plot_scatter(df, column)

# Cell 11 - Categorical data Analysis

ignored = [

    TARGET,

    "genres",

    "plot_keywords"

]

for column in cat_cols:

    if column not in ignored:

        plot_categorical(
            df,
            column
        )

# Cell 12 - Correlation Analysis
plt.figure(figsize=(15,10))

sns.heatmap(
    df[num_cols].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()