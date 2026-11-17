import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ============================================================
# FILE LOCATIONS
# ============================================================
# Paths are based on this file's location, so the code works no matter
# which folder it is started from.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "autoencoder.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.txt"

# Columns this module adds to its results. If an uploaded file already
# contains them (for example a file previously downloaded from the
# dashboard), the old values are dropped and recalculated.
OUTPUT_COLUMNS = ["reconstruction_error", "anomaly", "flag_reason", "Why Flagged"]


class ModelLoadError(RuntimeError):
    """The saved model, scaler or threshold could not be loaded."""


class InputDataError(ValueError):
    """The input data cannot be scored (missing columns, text, empty cells...)."""


# ============================================================
# LOAD TRAINED MODEL, SCALER AND THRESHOLD
# ============================================================

def load_model_components(model_path=None, scaler_path=None, threshold_path=None):
    """
    Load the trained autoencoder, scaler and anomaly threshold.

    Raises ModelLoadError with an understandable message if any file is
    missing or unreadable. Nothing is ever replaced by a default value.
    """

    model_path = Path(model_path or MODEL_PATH)
    scaler_path = Path(scaler_path or SCALER_PATH)
    threshold_path = Path(threshold_path or THRESHOLD_PATH)

    for label, path in [
        ("Autoencoder model", model_path),
        ("Scaler", scaler_path),
        ("Threshold", threshold_path),
    ]:
        if not path.exists():
            raise ModelLoadError(f"{label} file not found: {path}")

    try:
        model = load_model(model_path)
    except Exception as error:
        raise ModelLoadError(f"Could not load the Autoencoder model from {model_path}: {error}") from error

    try:
        scaler = joblib.load(scaler_path)
    except Exception as error:
        raise ModelLoadError(f"Could not load the scaler from {scaler_path}: {error}") from error

    try:
        threshold = float(threshold_path.read_text().strip())
    except Exception as error:
        raise ModelLoadError(f"Could not read a number from {threshold_path}: {error}") from error

    if not math.isfinite(threshold) or threshold <= 0:
        raise ModelLoadError(f"Threshold in {threshold_path} must be a positive number, got {threshold}")

    if not hasattr(scaler, "feature_names_in_"):
        raise ModelLoadError("The saved scaler does not record its feature names, so inputs cannot be checked.")

    return model, scaler, threshold


# ============================================================
# SELECT AND VALIDATE MODEL INPUT FEATURES
# ============================================================

def select_model_features(df, scaler):
    """
    Return exactly the columns the model was trained on, in training order.

    Returns:
        features        - DataFrame with only the model's input columns
        ignored_columns - list of extra columns that were present but not used

    Raises InputDataError with a plain-English message when the data
    cannot be scored.
    """

    required = list(scaler.feature_names_in_)

    if df is None or len(df) == 0:
        raise InputDataError("The file contains no transactions (0 rows).")

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise InputDataError(
            f"The file is missing {len(missing)} required column(s): {', '.join(missing)}. "
            f"The model needs these {len(required)} columns: {', '.join(required)}."
        )

    if df.columns.duplicated().any():
        duplicated = sorted(set(df.columns[df.columns.duplicated()]))
        raise InputDataError(f"The file has duplicated column names: {', '.join(duplicated)}.")

    features = df[required]

    non_numeric = [
        column for column in required
        if not pd.api.types.is_numeric_dtype(features[column])
        or pd.api.types.is_bool_dtype(features[column])
    ]
    if non_numeric:
        examples = []
        for column in non_numeric[:3]:
            converted = pd.to_numeric(features[column], errors="coerce")
            bad = features[column][converted.isna() & features[column].notna()]
            if len(bad):
                examples.append(f"{column} = '{bad.iloc[0]}' (row {bad.index[0]})")
        raise InputDataError(
            f"These columns must contain only numbers but contain text: {', '.join(non_numeric)}."
            + (f" Example: {'; '.join(examples)}." if examples else "")
        )

    empty_counts = features.isna().sum()
    if empty_counts.any():
        details = ", ".join(f"{column} ({count})" for column, count in empty_counts[empty_counts > 0].items())
        raise InputDataError(f"Some required columns have empty cells: {details}. Please fill or remove these rows.")

    infinite_counts = pd.Series(np.isinf(features.to_numpy(dtype=float)).sum(axis=0), index=required)
    if infinite_counts.any():
        details = ", ".join(f"{column} ({count})" for column, count in infinite_counts[infinite_counts > 0].items())
        raise InputDataError(f"Some required columns contain infinite values: {details}.")

    ignored_columns = [column for column in df.columns if column not in required]

    return features, ignored_columns


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

def detect_anomalies(df, components=None):
    """
    Detect anomalous transactions using the trained autoencoder.

    Args:
        df         - DataFrame of transactions. Extra columns (for example
                     Class, or columns from a previously downloaded results
                     file) are allowed and ignored by the model.
        components - optional (model, scaler, threshold) that are already
                     loaded. If omitted they are loaded from disk.

    Returns:
        DataFrame containing the original columns plus
        reconstruction error and anomaly prediction.
    """

    if components is None:
        components = load_model_components()

    model, scaler, threshold = components

    # --------------------------------------------------------
    # Select exactly the features used during training,
    # in the same order (raises InputDataError if impossible)
    # --------------------------------------------------------

    X, _ = select_model_features(df, scaler)

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
    # Create result DataFrame (old result columns are replaced)
    # --------------------------------------------------------

    results = df.drop(columns=[c for c in OUTPUT_COLUMNS if c in df.columns]).copy()

    results["reconstruction_error"] = reconstruction_error

    results["anomaly"] = predictions

    return results

# ============================================================
# TEST THE ANOMALY DETECTION
# ============================================================

if __name__ == "__main__":

    print("\n===== ANOMALY DETECTION =====")

    # Load dataset
    data_path = PROJECT_ROOT / "data" / "creditcard.csv"

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
    # NOTE: this runs on the FULL dataset, which includes the
    # rows the model was trained on. It is NOT a test result -
    # see src/baseline_evaluation.py for held-out evaluation.
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
