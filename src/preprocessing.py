import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/creditcard.csv"

def load_data():
    """Load the credit card transaction dataset."""
    return pd.read_csv(DATA_PATH)

def prepare_data(df):
    """
    Prepare transaction data for anomaly detection.

    Class labels are not used as model features.
    The Autoencoder is trained only on normal transactions.
    """

    # Separate features and target label
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Select only normal transactions for Autoencoder training
    X_normal = X[y == 0]

    # Create scaler and fit it ONLY on normal training data
    scaler = StandardScaler()

    X_normal_scaled = scaler.fit_transform(X_normal)

    # Transform the complete dataset using the same scaler
    X_all_scaled = scaler.transform(X)

    return X_normal_scaled, X_all_scaled, y, scaler

if __name__ == "__main__":

    df = load_data()

    X_normal_scaled, X_all_scaled, y, scaler = prepare_data(df)

    print("===== PREPROCESSING RESULTS =====")
    print("Original dataset shape:", df.shape)
    print("Features used:", X_all_scaled.shape[1])

    print(
        "Normal transactions used for Autoencoder training:",
        X_normal_scaled.shape[0]
    )

    print(
        "All transactions prepared for evaluation:",
        X_all_scaled.shape[0]
    )

    print("\nClass distribution:")
    print(y.value_counts())
    print("\nFeature scaling completed successfully.")