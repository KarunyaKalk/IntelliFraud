import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ============================================================
# LOAD TRAINED MODEL, SCALER AND THRESHOLD
# ============================================================

MODEL_PATH = "models/autoencoder.keras"
SCALER_PATH = "models/scaler.pkl"
THRESHOLD_PATH = "models/threshold.txt"

def load_model_components():
    """
    Load the trained autoencoder, scaler and anomaly threshold.
    """

    model = load_model(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    with open(THRESHOLD_PATH, "r") as file:
        threshold = float(file.read())

    return model, scaler, threshold

# ============================================================
# CALCULATE RECONSTRUCTION ERROR
# ============================================================

def calculate_reconstruction_error(model, X):
    """
    Calculate reconstruction error for each transaction.
    """

    reconstructed = model.predict(
        X,
        batch_size=512,
        verbose=0
    )

    errors = np.mean(
        np.square(X - reconstructed),
        axis=1
    )

    return errors

# ============================================================
# DETECT ANOMALIES
# ============================================================

def detect_anomalies(df):
    """
    Detect anomalous transactions using the trained autoencoder.

    Returns:
        DataFrame containing reconstruction error
        and anomaly prediction.
    """

    model, scaler, threshold = load_model_components()

    # --------------------------------------------------------
    # Select the same 30 features used during training
    # --------------------------------------------------------

    feature_columns = [
        column for column in df.columns
        if column != "Class"
    ]

    X = df[feature_columns]

    # --------------------------------------------------------
    # Scale the data using the saved scaler
    # --------------------------------------------------------

    X_scaled = scaler.transform(X)

    # --------------------------------------------------------
    # Calculate reconstruction error
    # --------------------------------------------------------

    reconstruction_error = calculate_reconstruction_error(
        model,
        X_scaled
    )

    # --------------------------------------------------------
    # Flag anomalies
    # --------------------------------------------------------

    predictions = (
        reconstruction_error > threshold
    ).astype(int)

    # --------------------------------------------------------
    # Create result DataFrame
    # --------------------------------------------------------

    results = df.copy()

    results["reconstruction_error"] = reconstruction_error

    results["anomaly"] = predictions

    return results

# ============================================================
# TEST THE ANOMALY DETECTION
# ============================================================

if __name__ == "__main__":

    print("\n===== ANOMALY DETECTION =====")

    # Load dataset
    data_path = "data/creditcard.csv"

    df = pd.read_csv(data_path)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # Detect anomalies
    results = detect_anomalies(df)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n===== DETECTION RESULTS =====")

    print(
        "Total transactions:",
        len(results)
    )

    print(
        "Anomalies detected:",
        results["anomaly"].sum()
    )

    print(
        "Normal transactions:",
        (results["anomaly"] == 0).sum()
    )

    # --------------------------------------------------------
    # Compare with actual fraud labels
    # --------------------------------------------------------

    if "Class" in results.columns:

        actual_fraud = results["Class"].sum()

        detected_fraud = (
            results[
                (results["Class"] == 1)
                &
                (results["anomaly"] == 1)
            ]
        ).shape[0]

        print(
            "Actual fraud cases:",
            actual_fraud
        )

        print(
            "Fraud cases detected:",
            detected_fraud
        )

    # --------------------------------------------------------
    # Show first 10 anomalies
    # --------------------------------------------------------

    print("\n===== FIRST 10 ANOMALIES =====")

    anomalies = results[
        results["anomaly"] == 1
    ]

    print(
        anomalies[
            [
                "Time",
                "Amount",
                "reconstruction_error",
                "anomaly"
            ]
        ].head(10)
    )

    print("\nAnomaly detection completed successfully.")