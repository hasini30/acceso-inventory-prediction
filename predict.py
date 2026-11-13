
import pandas as pd
import numpy as np
import joblib

# Load the trained model and preprocessing pipeline
model = joblib.load("best_model.joblib")

# Use the business-selected classification threshold
THRESHOLD = 0.35


# Function to predict kitchen transfer for one inventory batch
def score_batch(batch_dict: dict) -> dict:

    # Convert the raw inventory dictionary into a DataFrame
    batch = pd.DataFrame([batch_dict])

    # Convert date columns into datetime format
    batch["snapshot_date"] = pd.to_datetime(
        batch["snapshot_date"], errors="coerce"
    )

    batch["expiry_date"] = pd.to_datetime(
        batch["expiry_date"], errors="coerce"
    )

    # Convert unit cost into numeric format
    batch["unit_cost_inr"] = pd.to_numeric(
        batch["unit_cost_inr"]
        .astype(str)
        .str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce"
    )

    # Feature 1: Calculate remaining shelf life
    batch["remaining_shelf_life"] = (
        batch["expiry_date"] - batch["snapshot_date"]
    ).dt.days

    # Feature 2: Calculate net daily sales velocity
    batch["net_daily_sales_velocity"] = (
        (batch["last_3_days_sales"] / 3) *
        (1 - batch["doorstep_return_rate"])
    )

    # Feature 3: Estimate days required to clear stock
    batch["estimated_days_to_clear"] = np.where(
        batch["net_daily_sales_velocity"] > 0,
        batch["current_qty"] /
        batch["net_daily_sales_velocity"],
        np.nan
    )

    # Feature 4: Calculate stock depletion ratio
    batch["stock_depletion_ratio"] = (
        (batch["initial_qty"] - batch["current_qty"]) /
        batch["initial_qty"]
    )

    # Feature 5: Calculate expiry risk gap
    batch["expiry_risk_gap"] = (
        batch["estimated_days_to_clear"] -
        batch["remaining_shelf_life"]
    )

    # Select the same input columns used during model training
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

    # Generate probability of transfer to kitchen (class 1)
    probabilities = model.predict_proba(
        batch[feature_columns]
    )[0]

    # Find the probability corresponding to class 1
    class_one_index = list(model.classes_).index(1)
    risk_score = float(probabilities[class_one_index])

    # Apply the selected classification threshold
    prediction = int(risk_score >= THRESHOLD)

    # Return the required prediction result
    return {
        "batch_id": batch_dict["batch_id"],
        "transfer_to_kitchen": prediction,
        "expiry_risk_score": round(risk_score, 2)
    }


# Test the prediction function using one raw CSV record
if __name__ == "__main__":

    # Read the original dataset
    raw_data = pd.read_csv("accesco_trial_data.csv")

    # Select one raw inventory record
    sample_batch = raw_data.iloc[0].to_dict()

    # Generate the prediction
    result = score_batch(sample_batch)

    print("\n--- Inventory Batch Prediction ---")
    print(result)
