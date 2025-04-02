from contextlib import redirect_stdout
import io
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------
# 1. Load the Data
# ------------------------------
train_df = pd.read_csv("obesity_training.csv")
test_df = pd.read_csv("obesity_testing.csv")
sample_submission = pd.read_csv("sample_submission.csv")

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
# 7. Feature Engineering: Scaling & Polynomial Features
# ------------------------------
# Define the list of continuous columns (from the original data)
continuous_cols = ['Age', 'Height', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Scale the continuous features using StandardScaler
scaler = StandardScaler()
train_df_encoded[continuous_cols] = scaler.fit_transform(train_df_encoded[continuous_cols])
test_df_encoded[continuous_cols] = scaler.transform(test_df_encoded[continuous_cols])

# Generate polynomial features (degree 2: squared terms and pairwise interactions)
poly = PolynomialFeatures(degree=2, include_bias=False)
train_poly = poly.fit_transform(train_df_encoded[continuous_cols])
test_poly = poly.transform(test_df_encoded[continuous_cols])
poly_feature_names = poly.get_feature_names_out(continuous_cols)

# Create DataFrames for the polynomial features
train_poly_df = pd.DataFrame(train_poly, columns=poly_feature_names, index=train_df_encoded.index)
test_poly_df = pd.DataFrame(test_poly, columns=poly_feature_names, index=test_df_encoded.index)

# IMPORTANT: Drop the original continuous columns, so we keep ONLY the polynomial features
train_df_encoded.drop(columns=continuous_cols, inplace=True)
test_df_encoded.drop(columns=continuous_cols, inplace=True)

# Merge the polynomial features with the rest of the encoded data
train_df_final = pd.concat([train_df_encoded, train_poly_df], axis=1)
test_df_final = pd.concat([test_df_encoded, test_poly_df], axis=1)

# Double-check: Remove any lingering original continuous columns if present
train_df_final = train_df_final.drop(columns=continuous_cols, errors='ignore')
test_df_final = test_df_final.drop(columns=continuous_cols, errors='ignore')

# ------------------------------
# 8. Output File: Write Full Preprocessing and Feature Engineering Output to a File
# ------------------------------
with open("preprocessing_output_full.txt", "w") as f:
    with redirect_stdout(f):
        # Print initial shapes and column lists
        print("Initial Training Data Shape:", train_df.shape)
        print("Initial Testing Data Shape:", test_df.shape)
        print("\nNumeric Columns:", numeric_cols)
        print("Categorical Columns:", categorical_cols)

        # Print final training DataFrame info (after feature engineering)
        print("\nFinal Training Data (After Feature Engineering) Info:")
        buffer = io.StringIO()
        train_df_final.info(buf=buffer)
        print(buffer.getvalue())

        # Print final testing DataFrame info (after feature engineering)
        print("\nFinal Testing Data (After Feature Engineering) Info:")
        buffer = io.StringIO()
        test_df_final.info(buf=buffer)
        print(buffer.getvalue())

        # Print head of the final DataFrames
        print("\nHead of Final Training Data:")
        print(train_df_final.head())
        print("\nHead of Final Testing Data:")
        print(test_df_final.head())

# ------------------------------
# 9. Model Building and Evaluation with Hyperparameter Tuning
# ------------------------------
# Separate features (X) and target (y) from the final training DataFrame.
X = train_df_final.drop('Obesity', axis=1)
y = train_df_final['Obesity']

# Encode the target variable (since it's a multi-class string variable)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into a training set and a validation set (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define parameter grid for RandomForest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Setup GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid,
                           cv=3,
                           n_jobs=-1,
                           scoring='accuracy')

grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Use the best estimator found by GridSearchCV
best_clf = grid_search.best_estimator_

# Predict on the validation set and evaluate the performance
y_val_pred = best_clf.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy after tuning:", accuracy)
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

# ------------------------------
# 10. Generate Predictions for the Test Set and Prepare Submission
# ------------------------------
# For the test set, remove the "Obesity" column if it exists (test set is unlabeled)
if 'Obesity' in test_df_final.columns:
    X_test = test_df_final.drop('Obesity', axis=1)
else:
    X_test = test_df_final

# Predict using the tuned classifier
y_test_pred = best_clf.predict(X_test)

# Convert the predicted numeric labels back to the original class names
y_test_pred_labels = le.inverse_transform(y_test_pred)

# Prepare the submission file using sample_submission as a template.
submission = sample_submission.copy()
submission['Obesity'] = y_test_pred_labels
submission.to_csv("final_submission.csv", index=False)

print("Modeling completed and submission file 'final_submission.csv' generated!")
