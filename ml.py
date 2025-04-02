import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Load the datasets
train_df = pd.read_csv("obesity_training.csv")
test_df = pd.read_csv("obesity_testing.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# ------------------------------
# 1. Load the Data (Assuming data is already loaded)
# ------------------------------
# For example:
# train_df = pd.read_csv("obesity_training.csv")
# test_df = pd.read_csv("obesity_testing.csv")
# sample_submission = pd.read_csv("sample_submission.csv")

print("Initial Training Data Shape:", train_df.shape)
print("Initial Testing Data Shape:", test_df.shape)

# ------------------------------
# 2. Identify Numeric and Categorical Columns
# ------------------------------
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)

# ------------------------------
# 3. Handle Missing Values (Avoiding chained assignment warnings)
# ------------------------------

# For numeric columns (excluding 'ID'), fill missing values with the median
for col in numeric_cols:
    if col != 'ID':  # Skip ID column
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

# For categorical columns (excluding the target variable "Obesity"), fill missing values with the mode
for col in categorical_cols:
    if col != 'Obesity':
        mode_val = train_df[col].mode()[0]
        train_df[col] = train_df[col].fillna(mode_val)
        test_df[col] = test_df[col].fillna(mode_val)

# ------------------------------
# 4. Drop the ID Column (if not needed)
# ------------------------------
if 'ID' in train_df.columns:
    train_df.drop('ID', axis=1, inplace=True)
if 'ID' in test_df.columns:
    test_df.drop('ID', axis=1, inplace=True)

# ------------------------------
# 5. Encode Categorical Variables
# ------------------------------
# List of categorical columns to encode (excluding the target variable "Obesity")
cols_to_encode = [col for col in categorical_cols if col != 'Obesity']

train_df_encoded = pd.get_dummies(train_df, columns=cols_to_encode, drop_first=True)
test_df_encoded = pd.get_dummies(test_df, columns=cols_to_encode, drop_first=True)

# ------------------------------
# 6. Align the Training and Testing Datasets
# ------------------------------
train_df_encoded, test_df_encoded = train_df_encoded.align(test_df_encoded, join='left', axis=1, fill_value=0)

# ------------------------------
# 7. Review the Updated Datasets
# ------------------------------
print("\nUpdated Training Data (Encoded) Info:")
print(train_df_encoded.info())
print("\nUpdated Testing Data (Encoded) Info:")
print(test_df_encoded.info())

print("\nHead of Updated Training Data:")
print(train_df_encoded.head())
print("\nHead of Updated Testing Data:")
print(test_df_encoded.head())

