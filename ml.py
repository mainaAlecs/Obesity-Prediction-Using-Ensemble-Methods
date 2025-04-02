import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_df = pd.read_csv("obesity_training.csv")
test_df = pd.read_csv("obesity_testing.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Check shapes to know the number of records and features
# print("Training Data Shape:", train_df.shape)
# print("Testing Data Shape:", test_df.shape)

# Display first few rows of the training data
# print("First 5 rows of Training Data:")
# print(train_df.head())

# Get an overview of data types and non-null counts
# print("Data Information:")
# print(train_df.info())

print("Descriptive Statistics:")
print(train_df.describe())

# If there are categorical variables, review their frequency counts
for col in train_df.select_dtypes(include=['object']).columns:
    print(f"\nValue counts for {col}:")
    print(train_df[col].value_counts())
