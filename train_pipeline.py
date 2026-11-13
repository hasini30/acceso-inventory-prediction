
import pandas as pd
import numpy as np


# 1. LOAD THE RAW DATASET

# Reading the csv file and loading into pandas df
df = pd.read_csv("accesco_trial_data.csv")

# 2. INITIAL DATA INSPECTION


# Display column names, non-null counts, data types and memory usage
print("\n--- Dataset Information ---")
df.info()

# Count missing values in each column
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Check whether any batch IDs are duplicated
# duplicated() returns True for repeated batch IDs
print("\n--- Duplicate Batch IDs ---")
print(df["batch_id"].duplicated().sum())

# Count the number of records belonging to each target class
# 0 = Keep on grocery shelf
# 1 = Transfer to kitchen

print("\n--- Target Distribution ---")
print(df["transfer_to_kitchen"].value_counts())

# Display statistical information such as mean, standard deviation,
# minimum, maximum and quartiles for numerical columns
print("\n--- Statistical Summary ---")
print(df.describe())

# 3. DATA TYPE CONVERSION

# Convert snapshot_date from string to datetime format
df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"], errors="coerce"
)

# Convert expiry_date from string to datetime format

df["expiry_date"] = pd.to_datetime(
    df["expiry_date"], errors="coerce"
)

# Display the updated data types of both date columns
print("\n--- Date Conversion Completed ---")
print(df[["snapshot_date", "expiry_date"]].dtypes)


# Convert unit_cost_inr from text into numeric format
df["unit_cost_inr"] = pd.to_numeric(
    df["unit_cost_inr"]
    .astype(str)  # Convert values into strings for text processing
    .str.replace(
        r"[^0-9.]", "", regex=True
    ),  # Remove currency symbols and unwanted characters
    errors="coerce"  # Convert invalid numeric values into NaN
)

# Display the updated data type of unit_cost_inr
print("\n--- Unit Cost Data Type ---")
print(df["unit_cost_inr"].dtype)


# 4. HANDLING MISSING VALUES

# Fill missing sales values using the median sales
# calculated separately for each grocery category
df["last_3_days_sales"] = (
    df.groupby("category")["last_3_days_sales"]
    .transform(
        lambda x: x.fillna(x.median())
    )
)

# Check the number of missing values after imputation
print("\n--- Missing Values After Treatment ---")
print(df.isnull().sum())



# 5. IDENTIFYING DATA ANOMALIES


# Count records with negative current stock quantities
# Negative inventory quantities are impossible in this context
print("\n--- Negative Stock Records ---")
print((df["current_qty"] < 0).sum())


# Identify records where expiry date is earlier than snapshot date
# These records are invalid for our intended prediction scenario
print("\n--- Expiry Before Snapshot ---")
print((df["expiry_date"] < df["snapshot_date"]).sum())


# Identify records with impossible inventory quantities
# Initial quantity must be positive
# Current quantity cannot exceed initial quantity
print("\n--- Invalid Quantity Records ---")
print(
    (
        (df["initial_qty"] <= 0) |
        (df["current_qty"] > df["initial_qty"])
    ).sum()
)



# 6. REMOVING INVALID RECORDS

# Remove records where current stock quantity is negative
df = df[df["current_qty"] >= 0].copy()

# Remove records where expiry date is earlier than snapshot date
df = df[
    df["expiry_date"] >= df["snapshot_date"]
].copy()

# Remove records where initial quantity is zero or negative
# Also remove records where current quantity exceeds initial quantity
df = df[
    (df["initial_qty"] > 0) &
    (df["current_qty"] <= df["initial_qty"])
].copy()



# 7. FINAL DATA VALIDATION


# Display the number of rows and columns after cleaning
print("\n--- Final Dataset Shape ---")
print(df.shape)

# Check whether any missing values remain in the dataset
print("\n--- Remaining Missing Values ---")
print(df.isnull().sum())

# Verify that no negative stock quantities remain
print("\n--- Remaining Negative Stock ---")
print((df["current_qty"] < 0).sum())

# Save the cleaned dataset as a new CSV file
# The original raw dataset remains unchanged
df.to_csv("cleaned_inventory_data.csv", index=False)

print("\nData cleaning completed successfully!")





# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Feature 1: Calculate remaining shelf life in days
# Subtract snapshot_date from expiry_date to find
# how many days are left before the inventory expires.
# .dt.days extracts the difference as a number of days.

df["remaining_shelf_life"] = (
    df["expiry_date"] - df["snapshot_date"]
).dt.days

# Display batch IDs and their remaining shelf life
print("\n--- Remaining Shelf Life ---")
print(df[["batch_id", "remaining_shelf_life"]].head())


# Feature 2: Calculate net daily sales velocity
# last_3_days_sales represents total units sold in the last 3 days.
# Divide by 3 to calculate average daily sales.
# Multiply by (1 - doorstep_return_rate) to account
# for the proportion of sales not returned by customers.

df["net_daily_sales_velocity"] = (
    (df["last_3_days_sales"] / 3) *
    (1 - df["doorstep_return_rate"])
)


# Feature 3: Estimate the number of days required to clear stock
# Divide current stock quantity by net daily sales velocity.
# np.where checks whether sales velocity is greater than zero.
# If velocity is positive, calculate estimated clearing days.
# If velocity is zero or negative, assign NaN because
# clearing time cannot be estimated using this formula.

df["estimated_days_to_clear"] = np.where(
    df["net_daily_sales_velocity"] > 0,
    df["current_qty"] / df["net_daily_sales_velocity"],
    np.nan
)


# Feature 4: Calculate stock depletion ratio
# Measures the proportion of original inventory that has
# left the current stock.
# Subtract current quantity from initial quantity
# and divide by the initial quantity.
# A value of 0.60 means 60% of the original stock has depleted.

df["stock_depletion_ratio"] = (
    (df["initial_qty"] - df["current_qty"]) /
    df["initial_qty"]
)


# Feature 5: Calculate expiry risk gap
# Subtract remaining shelf life from estimated days to clear.
# A positive value indicates that estimated selling time
# is greater than the remaining shelf life.
# This helps identify inventory that may expire before selling.

df["expiry_risk_gap"] = (
    df["estimated_days_to_clear"] -
    df["remaining_shelf_life"]
)


# ==========================================
# DISPLAY ENGINEERED FEATURES
# ==========================================

# Display the original batch ID and all five new features
print("\n--- Feature Engineering Results ---")

print(df[[
    "batch_id",
    "remaining_shelf_life",
    "net_daily_sales_velocity",
    "estimated_days_to_clear",
    "stock_depletion_ratio",
    "expiry_risk_gap"
]].head())

# Display summary statistics of the newly created features
print("\n--- Feature Summary ---")

print(df[[
    "remaining_shelf_life",
    "net_daily_sales_velocity",
    "estimated_days_to_clear",
    "stock_depletion_ratio",
    "expiry_risk_gap"
]].describe())


# Check missing values in engineered features
print("\n--- Missing Values in New Features ---")

print(df[[
    "remaining_shelf_life",
    "net_daily_sales_velocity",
    "estimated_days_to_clear",
    "stock_depletion_ratio",
    "expiry_risk_gap"
]].isnull().sum())


print("\n--- Infinite Values ---")

print(
    np.isinf(
        df[[
            "remaining_shelf_life",
            "net_daily_sales_velocity",
            "estimated_days_to_clear",
            "stock_depletion_ratio",
            "expiry_risk_gap"
        ]].select_dtypes(include="number")
    ).sum()
)

#Model training and comparison
# Select input features
feature_columns = [
    "category",
    "initial_qty",
    "current_qty",
    "last_3_days_sales",
    "doorstep_return_rate",
    "unit_cost_inr",
    "remaining_shelf_life",
    "net_daily_sales_velocity",
    "estimated_days_to_clear",
    "stock_depletion_ratio",
    "expiry_risk_gap"
]

# X contains the input features
X = df[feature_columns]

# y contains the target variable
y = df["transfer_to_kitchen"]

# split the dataset into training and testing sets using stratified sampling

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display the number of training and testing records
print("\n--- Dataset Split ---")
print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# Identify numerical and categorical columns
numeric_features = [
    col for col in feature_columns
    if col != "category"
]

categorical_features = ["category"]


# Preprocess numerical features
numeric_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Preprocess categorical features
categorical_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_preprocessor, numeric_features),
    ("cat", categorical_preprocessor, categorical_features)
])

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Create Logistic Regression pipeline
logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model using training data
logistic_model.fit(X_train, y_train)

print("\nLogistic Regression training completed!")

# Create Random Forest pipeline
random_forest_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train the model using training data
random_forest_model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Function to evaluate model performance

def evaluate_model(model, X_test, y_test, model_name):

    # Predict target classes
    y_pred = model.predict(X_test)

    # Predict probability of class 1
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Display model evaluation results
    print(f"\n--- {model_name} Evaluation ---")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("ROC-AUC:", roc_auc)

    return f1

# Evaluate Logistic Regression
lr_f1 = evaluate_model(
    logistic_model,
    X_test,
    y_test,
    "Logistic Regression"
)

# Evaluate Random Forest
rf_f1 = evaluate_model(
    random_forest_model,
    X_test,
    y_test,
    "Random Forest"
)

import joblib

# Select the model with the higher F1-score
if rf_f1 > lr_f1:
    best_model = random_forest_model
    best_model_name = "Random Forest"
else:
    best_model = logistic_model
    best_model_name = "Logistic Regression"

# Save the complete pipeline, including preprocessing
joblib.dump(best_model, "best_model.joblib")

print("\nBest Model:", best_model_name)
print("Model saved successfully!")

#threshold adjustment



from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

# Get predicted probabilities for class 1
y_prob = best_model.predict_proba(X_test)[:, 1]

# Set a lower classification threshold
threshold = 0.35

# Classify batches using the adjusted threshold
y_pred_adjusted = (y_prob >= threshold).astype(int)

# Evaluate the adjusted predictions
print("\n--- Adjusted Threshold Evaluation ---")

print("Threshold:", threshold)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_adjusted))

print("Precision:", precision_score(y_test, y_pred_adjusted))
print("Recall:", recall_score(y_test, y_pred_adjusted))
print("F1-Score:", f1_score(y_test, y_pred_adjusted))
