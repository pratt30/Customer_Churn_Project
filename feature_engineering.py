import pandas as pd
import numpy as np


service_cols = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]


def feature_engineering(frame):

    x = frame.copy()

    # Convert TotalCharges to numeric
    x["TotalCharges"] = pd.to_numeric(
        x["TotalCharges"],
        errors="coerce"
    )

    # Average monthly charge
    x["avg_monthly_charge"] = np.where(
        x["tenure"] > 0,
        x["TotalCharges"] / x["tenure"],
        x["MonthlyCharges"]
    )

    # Count subscribed services
    x["service_count"] = sum(
        (x[c] == "Yes").astype(int)
        for c in service_cols
    )

    # Customer ID is not used for prediction
    if "customerID" in x.columns:
        x = x.drop(columns=["customerID"])

    return x