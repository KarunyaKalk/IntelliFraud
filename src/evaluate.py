import os
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from anomaly_detection import detect_anomalies

DATA_PATH = "data/creditcard.csv"
TEST_INDEX_PATH = "models/test_indices.csv"
RESULT_PATH = "models/evaluation_results.csv"

print("\n===== HELD-OUT TEST EVALUATION =====")

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
print("Held-out normal cases:", (test_df["Class"] == 0).sum())

# ---------------------------------------------------------
# 3. Run anomaly detection on HELD-OUT TEST SET
# ---------------------------------------------------------
results = detect_anomalies(test_df)

y_true = results["Class"]
y_pred = results["anomaly"]

# ---------------------------------------------------------
# 4. Calculate evaluation metrics
# ---------------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

# ---------------------------------------------------------
# 5. Confusion matrix
# ---------------------------------------------------------
tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred
).ravel()

# ---------------------------------------------------------
# 6. Print results
# ---------------------------------------------------------
print("\n===== EVALUATION RESULTS =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n===== CONFUSION MATRIX =====")

print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

print("\nActual fraud cases:", y_true.sum())
print("Fraud cases caught:", tp)
print("Fraud cases missed:", fn)

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)

# ---------------------------------------------------------
# 7. Save evaluation results
# ---------------------------------------------------------
evaluation_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives",
        "Actual Fraud Cases",
        "Fraud Cases Caught",
        "Fraud Cases Missed"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        tn,
        fp,
        fn,
        tp,
        int(y_true.sum()),
        tp,
        fn
    ]
})

evaluation_results.to_csv(
    RESULT_PATH,
    index=False
)

print("\nEvaluation results saved to:", RESULT_PATH)

print("\n===== HELD-OUT EVALUATION COMPLETED =====")