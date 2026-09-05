import os
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from anomaly_detection import (
    load_model_components,
    calculate_reconstruction_error
)

DATA_PATH = "data/creditcard.csv"
TEST_INDEX_PATH = "models/test_indices.csv"
RESULT_PATH = "models/model_comparison.csv"

print("\n===== MODEL COMPARISON ON HELD-OUT TEST SET =====")

# ---------------------------------------------------------
# 1. Load complete dataset
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("Complete dataset:", df.shape)

# ---------------------------------------------------------
# 2. Load held-out test indices
# ---------------------------------------------------------
if not os.path.exists(TEST_INDEX_PATH):
    raise FileNotFoundError(
        "test_indices.csv not found. "
        "Please run train.py first."
    )

test_indices = pd.read_csv(TEST_INDEX_PATH)["index"].astype(int)

test_df = df.loc[test_indices].copy()

print("Held-out test transactions:", len(test_df))
print("Held-out fraud cases:", test_df["Class"].sum())

# ---------------------------------------------------------
# 3. Prepare training-normal and test data
# ---------------------------------------------------------
train_indices = df.index.difference(test_indices)

train_df = df.loc[train_indices].copy()

train_normal = train_df[
    train_df["Class"] == 0
].copy()

feature_columns = [
    column for column in df.columns
    if column != "Class"
]

X_train_normal = train_normal[feature_columns]
X_test = test_df[feature_columns]

y_test = test_df["Class"]

# ---------------------------------------------------------
# 4. Load the scaler
# ---------------------------------------------------------
_, scaler, _ = load_model_components()

X_train_normal_scaled = scaler.transform(
    X_train_normal
)

X_test_scaled = scaler.transform(
    X_test
)

# ---------------------------------------------------------
# 5. Autoencoder
# ---------------------------------------------------------
print("\n===== AUTOENCODER =====")

model, _, threshold = load_model_components()

ae_errors = calculate_reconstruction_error(
    model,
    X_test_scaled
)

ae_predictions = (
    ae_errors > threshold
).astype(int)

# ---------------------------------------------------------
# 6. Isolation Forest
# ---------------------------------------------------------
print("\n===== ISOLATION FOREST =====")

isolation_forest = IsolationForest(
    n_estimators=100,
    max_samples=10000,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

isolation_forest.fit(
    X_train_normal_scaled
)

if_predictions_raw = isolation_forest.predict(
    X_test_scaled
)

if_predictions = (
    if_predictions_raw == -1
).astype(int)

# ---------------------------------------------------------
# 7. One-Class SVM
# ---------------------------------------------------------
print("\n===== ONE-CLASS SVM =====")

sample_size = min(
    3000,
    len(X_train_normal_scaled)
)

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    len(X_train_normal_scaled),
    size=sample_size,
    replace=False
)

X_ocsvm_train = X_train_normal_scaled[
    sample_indices
]

one_class_svm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05
)

one_class_svm.fit(
    X_ocsvm_train
)

ocsvm_predictions_raw = one_class_svm.predict(
    X_test_scaled
)

ocsvm_predictions = (
    ocsvm_predictions_raw == -1
).astype(int)

# ---------------------------------------------------------
# 8. Evaluation function
# ---------------------------------------------------------
def evaluate_model(name, y_true, predictions):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    fraud_caught = int(
        ((y_true == 1) & (predictions == 1)).sum()
    )

    total_fraud = int(
        (y_true == 1).sum()
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Fraud Caught": fraud_caught,
        "Total Fraud": total_fraud
    }

# ---------------------------------------------------------
# 9. Compare all models
# ---------------------------------------------------------
results = []

results.append(
    evaluate_model(
        "Autoencoder",
        y_test,
        ae_predictions
    )
)

results.append(
    evaluate_model(
        "Isolation Forest",
        y_test,
        if_predictions
    )
)

results.append(
    evaluate_model(
        "One-Class SVM",
        y_test,
        ocsvm_predictions
    )
)

comparison_df = pd.DataFrame(results)

# ---------------------------------------------------------
# 10. Display results
# ---------------------------------------------------------
print("\n===== FINAL MODEL COMPARISON =====")

print(
    comparison_df.to_string(
        index=False
    )
)

# ---------------------------------------------------------
# 11. Save results
# ---------------------------------------------------------
comparison_df.to_csv(
    RESULT_PATH,
    index=False
)

print(
    "\nComparison results saved to:",
    RESULT_PATH
)

print(
    "\n===== MODEL COMPARISON COMPLETED ====="
)