"""
Baseline evaluation of the ORIGINAL (pre-existing) IntelliFraud Autoencoder.

This script is strictly READ-ONLY with respect to the model and the data:
  * it loads models/autoencoder.keras, models/scaler.pkl, models/threshold.txt
    and models/test_indices.csv, and data/creditcard.csv - none are modified;
  * it does NOT retrain, tune, or re-select the threshold;
  * the test set is never used to choose anything - the original threshold
    (95th percentile of training-normal errors) is used unchanged.

Outputs (new files only, all prefixed "baseline_"):
  results/metrics/baseline_evaluation_report.txt
  results/metrics/baseline_metrics.csv
  results/metrics/baseline_test_scores.csv
  results/graphs/baseline_pr_curve.png
  results/graphs/baseline_reconstruction_error_distribution.png
  results/graphs/baseline_confusion_matrix.png

Official baseline   = the full held-out test set (56,962 rows), as originally used.
Sensitivity analysis = the same test set without rows that have an exact copy
                       in the training set (reported separately, not official).

Safety checks (the script stops with an error if any fails):
  * re-created split must equal models/test_indices.csv;
  * reproduced confusion matrix must equal models/evaluation_results.csv;
  * recomputed threshold must match models/threshold.txt within
    THRESHOLD_REL_TOLERANCE;
  * protected input files must be byte-identical before and after the run.

Confidence intervals use a STRATIFIED bootstrap (class counts preserved).
Train/test duplicates are found by hash and confirmed by full-row equality.

Run from anywhere (the script switches to the project root):
    python src/baseline_evaluation.py
"""

import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)  # the original src/anomaly_detection.py uses relative paths
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.anomaly_detection import (
    calculate_reconstruction_error,
    detect_anomalies,
    load_model_components,
)

# ------------------------------------------------------------------
# Settings (identical to the original train.py split)
# ------------------------------------------------------------------
DATA_PATH = Path("data/creditcard.csv")
TEST_INDEX_PATH = Path("models/test_indices.csv")
ORIGINAL_EVAL_PATH = Path("models/evaluation_results.csv")
PROTECTED_FILES = [
    DATA_PATH,
    Path("models/autoencoder.keras"),
    Path("models/scaler.pkl"),
    Path("models/threshold.txt"),
    TEST_INDEX_PATH,
    ORIGINAL_EVAL_PATH,
    Path("models/model_comparison.csv"),
    Path("models/training_history.csv"),
]
SPLIT_TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 42
THRESHOLD_PERCENTILE = 95
# Maximum allowed relative difference between the saved threshold and the one
# recomputed from training-normal errors. The model outputs float32 numbers
# (about 7 significant digits), so tiny rounding differences are expected;
# anything above 1e-5 (0.001%) means the threshold cannot be trusted.
THRESHOLD_REL_TOLERANCE = 1e-5

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
CI_LEVEL = 95

METRIC_DIR = Path("results/metrics")
GRAPH_DIR = Path("results/graphs")
REPORT_PATH = METRIC_DIR / "baseline_evaluation_report.txt"
METRICS_CSV_PATH = METRIC_DIR / "baseline_metrics.csv"
SCORES_CSV_PATH = METRIC_DIR / "baseline_test_scores.csv"
PR_CURVE_PATH = GRAPH_DIR / "baseline_pr_curve.png"
ERROR_DIST_PATH = GRAPH_DIR / "baseline_reconstruction_error_distribution.png"
CONFUSION_PATH = GRAPH_DIR / "baseline_confusion_matrix.png"

LEGIT_COLOR = "#4C72B0"
FRAUD_COLOR = "#D1495B"
NEUTRAL_COLOR = "#555555"


class Report:
    """Prints text and keeps it for the saved report."""

    def __init__(self):
        self.lines = []

    def add(self, text=""):
        print(text)
        self.lines.append(str(text))

    def section(self, title):
        self.add()
        self.add("=" * 74)
        self.add(title)
        self.add("=" * 74)

    def save(self, path):
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_metrics(y_true, scores, y_pred):
    """All metrics for one evaluation set. scores = reconstruction errors."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    curve_precision, curve_recall, _ = precision_recall_curve(y_true, scores)
    return {
        "test_rows": int(len(y_true)),
        "actual_fraud": int(y_true.sum()),
        "actual_normal": int((y_true == 0).sum()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "accuracy": (tp + tn) / len(y_true),
        "always_normal_accuracy": (y_true == 0).mean(),
        "average_precision": average_precision_score(y_true, scores),
        "pr_auc_trapezoidal": auc(curve_recall, curve_precision),
        "roc_auc": roc_auc_score(y_true, scores),
        "no_skill_pr_auc": y_true.mean(),
        "false_alarms_per_fraud_caught": fp / tp if tp else float("nan"),
    }


BOOTSTRAP_METRICS = ["precision", "recall", "f1_score", "average_precision", "pr_auc_trapezoidal", "roc_auc"]


def bootstrap_ci(y_true, scores, y_pred):
    """
    STRATIFIED percentile bootstrap.

    In every one of the N_BOOTSTRAP resamples, legitimate rows are drawn (with
    replacement) only from the legitimate test rows, and fraud rows only from
    the fraud test rows. Each resample therefore has exactly the same number of
    legitimate (56,864) and fraud (98) rows as the real test set - only WHICH
    rows are picked changes. The threshold is never changed.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    y_true, scores, y_pred = map(np.asarray, (y_true, scores, y_pred))
    normal_rows = np.flatnonzero(y_true == 0)
    fraud_rows = np.flatnonzero(y_true == 1)
    samples = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate([
            rng.choice(normal_rows, size=len(normal_rows), replace=True),
            rng.choice(fraud_rows, size=len(fraud_rows), replace=True),
        ])
        yt, sc, yp = y_true[idx], scores[idx], y_pred[idx]
        assert int(yt.sum()) == len(fraud_rows) and len(yt) == len(y_true)
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        samples["precision"].append(p)
        samples["recall"].append(r)
        samples["f1_score"].append(2 * p * r / (p + r) if (p + r) else 0.0)
        samples["average_precision"].append(average_precision_score(yt, sc))
        cp, cr, _ = precision_recall_curve(yt, sc)
        samples["pr_auc_trapezoidal"].append(auc(cr, cp))
        samples["roc_auc"].append(roc_auc_score(yt, sc))
    alpha = (100 - CI_LEVEL) / 2
    return {
        name: (float(np.percentile(vals, alpha)), float(np.percentile(vals, 100 - alpha)))
        for name, vals in samples.items()
    }


def find_verified_duplicates(df, row_hash, test_index, reference_index):
    """
    For each test row, is there a row in `reference_index` that is EXACTLY equal?

    Step 1 (fast): compare 64-bit row hashes to find candidate matches.
    Step 2 (exact): for every candidate, compare all column values of the two
    rows directly. Only rows that pass step 2 count as duplicates, so a hash
    collision (two different rows with the same hash) can never be miscounted.
    Equal rows always have equal hashes, so step 1 cannot miss a true duplicate.

    Returns (verified flags, number of hash candidates, number rejected).
    """
    values = df.to_numpy()
    position = pd.Series(np.arange(len(df)), index=df.index)
    reference_by_hash = {}
    for idx, h in row_hash.loc[reference_index].items():
        reference_by_hash.setdefault(h, []).append(position[idx])

    verified = np.zeros(len(test_index), dtype=bool)
    candidates = rejected = 0
    for i, (idx, h) in enumerate(row_hash.loc[test_index].items()):
        matches = reference_by_hash.get(h)
        if not matches:
            continue
        candidates += 1
        test_values = values[position[idx]]
        if any(np.array_equal(test_values, values[j]) for j in matches):
            verified[i] = True
        else:
            rejected += 1
    return verified, candidates, rejected


def main():
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    report = Report()
    report.add("IntelliFraud - BASELINE evaluation of the ORIGINAL Autoencoder")
    report.add("(pre-existing model, evaluated read-only; no retraining, no tuning)")

    hashes_before = {str(p): sha256_of(p) for p in PROTECTED_FILES}
    report.section("0. INPUT FILES (SHA-256 fingerprints)")
    for path, digest in hashes_before.items():
        report.add(f"{path:<34} {digest}")

    # ==============================================================
    # 1. Load data and the saved split
    # ==============================================================
    df = pd.read_csv(DATA_PATH)
    feature_columns = [c for c in df.columns if c != "Class"]
    saved_test_index = pd.read_csv(TEST_INDEX_PATH)["index"].astype(int)

    # Re-create train.py's split in memory to identify the training rows.
    train_df, test_split_df = train_test_split(
        df, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_RANDOM_STATE, stratify=df["Class"]
    )
    split_matches = bool((test_split_df.index.values == saved_test_index.values).all())
    if not split_matches:
        raise RuntimeError("Re-created split does not match models/test_indices.csv - stopping.")
    test_df = df.loc[saved_test_index].copy()
    train_normal = train_df[train_df["Class"] == 0]

    # ==============================================================
    # 2. STEP A - reproduce the ORIGINAL evaluation and verify it
    # ==============================================================
    report.section("1. VERIFICATION - reproduce the original evaluation (src/evaluate.py logic)")
    results = detect_anomalies(test_df)  # original function, original threshold
    y_true = results["Class"].to_numpy()
    scores = results["reconstruction_error"].to_numpy()
    y_pred = results["anomaly"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    stored = pd.read_csv(ORIGINAL_EVAL_PATH).set_index("Metric")["Value"]
    expected = {
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp,
    }
    all_match = True
    for name, value in expected.items():
        ok = int(stored[name]) == int(value)
        all_match &= ok
        report.add(f"{name:<16} stored={int(stored[name]):>6,}  reproduced={int(value):>6,}  "
                   f"{'MATCH' if ok else 'MISMATCH'}")
    report.add(f"Split re-created from train.py settings matches test_indices.csv: {split_matches}")
    if not all_match:
        report.save(REPORT_PATH)
        raise RuntimeError("Reproduced confusion matrix differs from models/evaluation_results.csv - stopping.")
    report.add("RESULT: original confusion matrix verified - proceeding.")

    # ==============================================================
    # 3. STEP B - verify the threshold provenance (training data only)
    # ==============================================================
    report.section("2. VERIFICATION - threshold provenance (training-normal rows only)")
    model, scaler, threshold = load_model_components()
    train_normal_errors = calculate_reconstruction_error(
        model, scaler.transform(train_normal[feature_columns])
    )
    recomputed_threshold = float(np.percentile(train_normal_errors, THRESHOLD_PERCENTILE))
    rel_diff = abs(recomputed_threshold - threshold) / threshold
    report.add(f"Training-normal rows scored        : {len(train_normal):,} (fraud rows used: 0)")
    report.add(f"Saved threshold (threshold.txt)    : {threshold:.10f}")
    report.add(f"Recomputed {THRESHOLD_PERCENTILE}th percentile        : {recomputed_threshold:.10f}")
    report.add(f"Relative difference                : {rel_diff:.2e}")
    report.add(f"Allowed relative tolerance         : {THRESHOLD_REL_TOLERANCE:.0e}")
    if not rel_diff <= THRESHOLD_REL_TOLERANCE:  # also catches NaN
        report.add("RESULT: FAILED - threshold NOT reproduced. Stopping before computing any metrics.")
        report.save(REPORT_PATH)
        raise RuntimeError(
            f"Threshold check failed: saved threshold {threshold:.10f} vs recomputed "
            f"{recomputed_threshold:.10f} (relative difference {rel_diff:.2e} > tolerance "
            f"{THRESHOLD_REL_TOLERANCE:.0e}). The saved model, scaler and threshold may not belong "
            f"together - evaluation stopped."
        )
    report.add("RESULT: threshold reproduced from training-normal errors within tolerance. "
               "Test data played no part.")
    report.add(f"Share of training-normal rows above threshold: "
               f"{(train_normal_errors > threshold).mean():.2%} (by design ~{100 - THRESHOLD_PERCENTILE}%)")

    # ==============================================================
    # 4. Duplicate overlap flag for every test row
    # ==============================================================
    row_hash = pd.util.hash_pandas_object(df, index=False)
    duplicate_overlap, dup_candidates, dup_rejected = find_verified_duplicates(
        df, row_hash, test_df.index, train_df.index
    )
    seen_by_autoencoder, seen_candidates, seen_rejected = find_verified_duplicates(
        df, row_hash, test_df.index, train_normal.index
    )

    # ==============================================================
    # 5. Metrics - OFFICIAL (full test set) and SENSITIVITY
    # ==============================================================
    official = compute_metrics(y_true, scores, y_pred)
    keep = ~duplicate_overlap
    sensitivity = compute_metrics(y_true[keep], scores[keep], y_pred[keep])
    print(f"\nRunning {N_BOOTSTRAP} bootstrap resamples (seed {BOOTSTRAP_SEED})...")
    ci = bootstrap_ci(y_true, scores, y_pred)

    def pct(x):
        return f"{x * 100:.2f}%"

    def ci_text(name):
        low, high = ci[name]
        return f"[{pct(low)} - {pct(high)}]"

    report.section("3. OFFICIAL BASELINE - full held-out test set, original threshold")
    m = official
    report.add(f"Test transactions : {m['test_rows']:,}  (normal {m['actual_normal']:,}, fraud {m['actual_fraud']:,})")
    report.add(f"Threshold         : {threshold:.4f} (unchanged)")
    report.add()
    report.add("Confusion matrix (rows = actual, columns = predicted):")
    report.add(f"                    Pred. normal   Pred. fraud")
    report.add(f"  Actual normal     {m['true_negatives']:>12,}   {m['false_positives']:>11,}")
    report.add(f"  Actual fraud      {m['false_negatives']:>12,}   {m['true_positives']:>11,}")
    report.add()
    report.add(f"{'Metric':<34}{'Value':>10}   {CI_LEVEL}% bootstrap CI")
    report.add(f"{'Precision':<34}{pct(m['precision']):>10}   {ci_text('precision')}")
    report.add(f"{'Recall (fraud caught)':<34}{pct(m['recall']):>10}   {ci_text('recall')}")
    report.add(f"{'F1-score':<34}{pct(m['f1_score']):>10}   {ci_text('f1_score')}")
    report.add(f"{'Average Precision (AP)':<34}{m['average_precision']:>10.4f}   "
               f"[{ci['average_precision'][0]:.4f} - {ci['average_precision'][1]:.4f}]")
    report.add(f"{'PR-AUC (trapezoidal)':<34}{m['pr_auc_trapezoidal']:>10.4f}   "
               f"[{ci['pr_auc_trapezoidal'][0]:.4f} - {ci['pr_auc_trapezoidal'][1]:.4f}]")
    report.add(f"{'ROC-AUC':<34}{m['roc_auc']:>10.4f}   "
               f"[{ci['roc_auc'][0]:.4f} - {ci['roc_auc'][1]:.4f}]")
    report.add(f"{'Specificity (normal kept normal)':<34}{pct(m['specificity']):>10}")
    report.add(f"{'False-positive rate':<34}{pct(m['false_positive_rate']):>10}")
    report.add(f"{'Accuracy':<34}{pct(m['accuracy']):>10}")
    report.add(f"{'Accuracy of always saying normal':<34}{pct(m['always_normal_accuracy']):>10}")
    report.add(f"{'No-skill PR-AUC (= fraud rate)':<34}{m['no_skill_pr_auc']:>10.4f}")
    report.add(f"{'False alarms per fraud caught':<34}{m['false_alarms_per_fraud_caught']:>10.1f}")
    report.add()
    report.add("Confidence-interval method (stratified percentile bootstrap):")
    report.add(f"  - {N_BOOTSTRAP:,} resamples, random seed {BOOTSTRAP_SEED}.")
    report.add(f"  - In each resample, {m['actual_normal']:,} legitimate rows are drawn with replacement from the "
               f"legitimate test rows")
    report.add(f"    and {m['actual_fraud']} fraud rows are drawn with replacement from the fraud test rows, so every "
               f"resample keeps")
    report.add("    the original class counts (and fraud rate) exactly.")
    alpha = (100 - CI_LEVEL) / 2
    report.add(f"  - Metrics are recomputed on each resample with the threshold held fixed; the {CI_LEVEL}% CI is the")
    report.add(f"    {alpha:g}th-{100 - alpha:g}th percentile range of the {N_BOOTSTRAP:,} values. "
               f"Point estimates use the real test set.")
    report.add("  - The CI reflects which rows happened to land in the test set; it does not capture retraining")
    report.add("    variability (different weights/threshold), which this read-only evaluation cannot measure.")

    report.section("4. SENSITIVITY ANALYSIS (NOT the official baseline) - duplicates removed")
    s = sensitivity
    removed = int(duplicate_overlap.sum())
    report.add(f"Removed test rows with an exact copy in the training set: {removed:,} "
               f"(normal {int((duplicate_overlap & (y_true == 0)).sum()):,}, "
               f"fraud {int((duplicate_overlap & (y_true == 1)).sum()):,})")
    report.add(f"  of which seen by the Autoencoder (copy in training-normal): {int(seen_by_autoencoder.sum()):,}")
    report.add(f"Remaining: {s['test_rows']:,} rows (normal {s['actual_normal']:,}, fraud {s['actual_fraud']:,})")
    report.add()
    report.add("Duplicate detection method (all 31 columns, including Time, Amount and Class):")
    report.add("  Step 1 - row hashes find candidate matches; Step 2 - every candidate is confirmed by")
    report.add("  comparing all column values directly. Only confirmed matches count as duplicates.")
    report.add(f"  vs whole training set : {dup_candidates:,} hash candidates, "
               f"{dup_candidates - dup_rejected:,} confirmed, {dup_rejected:,} rejected (hash collisions)")
    report.add(f"  vs training-normal set: {seen_candidates:,} hash candidates, "
               f"{seen_candidates - seen_rejected:,} confirmed, {seen_rejected:,} rejected (hash collisions)")
    report.add()
    report.add(f"{'Metric':<28}{'Official':>12}{'Dup-excluded':>14}")
    for label, key, fmt in [
        ("True positives", "true_positives", "{:,}"),
        ("False positives", "false_positives", "{:,}"),
        ("False negatives", "false_negatives", "{:,}"),
        ("True negatives", "true_negatives", "{:,}"),
        ("Precision", "precision", "pct"),
        ("Recall", "recall", "pct"),
        ("F1-score", "f1_score", "pct"),
        ("False-positive rate", "false_positive_rate", "pct"),
        ("Average Precision", "average_precision", "{:.4f}"),
        ("PR-AUC (trapezoidal)", "pr_auc_trapezoidal", "{:.4f}"),
        ("ROC-AUC", "roc_auc", "{:.4f}"),
    ]:
        f = pct if fmt == "pct" else fmt.format
        report.add(f"{label:<28}{f(m[key]):>12}{f(s[key]):>14}")
    report.add()
    report.add("Duplicated normal test rows the Autoencoder had already seen:")
    seen_scores = scores[seen_by_autoencoder]
    report.add(f"  flagged as fraud: {int((seen_scores > threshold).sum()):,} of {len(seen_scores):,} "
               f"| median error {np.median(seen_scores):.4f} vs median of all normal test rows "
               f"{np.median(scores[y_true == 0]):.4f}")
    report.add(f"Duplicate copies inside the test set itself (extra copies): {int(test_df.duplicated().sum()):,}")
    removed_fraud = duplicate_overlap & (y_true == 1)
    direction = "LOWER" if s["average_precision"] < m["average_precision"] else "HIGHER"
    report.add()
    report.add(f"Interpretation: without the overlapping rows, Average Precision is {direction} "
               f"({s['average_precision']:.4f} vs {m['average_precision']:.4f}). "
               f"{int((removed_fraud & (y_pred == 1)).sum())} of the {int(removed_fraud.sum())} removed frauds "
               f"had been caught, and {int((seen_by_autoencoder & (y_pred == 1)).sum())} of the "
               f"{int(seen_by_autoencoder.sum())} removed normal rows had been false alarms.")

    # ==============================================================
    # 6. Reconstruction-error summary
    # ==============================================================
    report.section("5. RECONSTRUCTION ERROR SUMMARY (test set)")
    summary = pd.DataFrame({
        "Normal": pd.Series(scores[y_true == 0]).describe(percentiles=[0.5, 0.95, 0.99]),
        "Fraud": pd.Series(scores[y_true == 1]).describe(percentiles=[0.5, 0.95, 0.99]),
    })
    report.add(summary.round(4).to_string())
    report.add(f"Normal test rows above threshold: {pct((scores[y_true == 0] > threshold).mean())}")
    report.add(f"Fraud test rows above threshold : {pct((scores[y_true == 1] > threshold).mean())}")

    # ==============================================================
    # 7. Save tables
    # ==============================================================
    rows = []
    for set_name, metrics in [("official_full_test", official), ("sensitivity_duplicates_excluded", sensitivity)]:
        for key, value in metrics.items():
            low, high = ci.get(key, (np.nan, np.nan)) if set_name == "official_full_test" else (np.nan, np.nan)
            rows.append({"evaluation_set": set_name, "metric": key, "value": value,
                         "ci95_low": low, "ci95_high": high})
    rows.append({"evaluation_set": "official_full_test", "metric": "threshold", "value": threshold,
                 "ci95_low": np.nan, "ci95_high": np.nan})
    pd.DataFrame(rows).to_csv(METRICS_CSV_PATH, index=False)

    pd.DataFrame({
        "test_row_index": test_df.index.to_numpy(),
        "actual_class": y_true.astype(int),
        "reconstruction_error": scores,
        "predicted_class": y_pred.astype(int),
        "duplicate_overlap": duplicate_overlap,
    }).to_csv(SCORES_CSV_PATH, index=False)

    # ==============================================================
    # 8. Graphs
    # ==============================================================
    # 8a. Precision-recall curve
    curve_p, curve_r, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(curve_r, curve_p, color=LEGIT_COLOR, lw=2,
            label=f"Autoencoder (AP = {official['average_precision']:.4f}, "
                  f"PR-AUC = {official['pr_auc_trapezoidal']:.4f})")
    ax.axhline(official["no_skill_pr_auc"], color=NEUTRAL_COLOR, ls="--", lw=1,
               label=f"Random guessing (= fraud rate {official['no_skill_pr_auc']:.4f})")
    ax.scatter([official["recall"]], [official["precision"]], color=FRAUD_COLOR, zorder=5, s=60,
               label=f"Current threshold {threshold:.4f}\n(recall {pct(official['recall'])}, "
                     f"precision {pct(official['precision'])})")
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title("Baseline Autoencoder - Precision-Recall curve (held-out test set)")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PR_CURVE_PATH, dpi=150)
    plt.close(fig)

    # 8b. Reconstruction-error distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(np.log10(scores.min()), np.log10(scores.max()), 70)
    ax.hist(scores[y_true == 0], bins=bins, density=True, alpha=0.55, color=LEGIT_COLOR,
            label=f"Normal (n={official['actual_normal']:,})")
    ax.hist(scores[y_true == 1], bins=bins, density=True, alpha=0.6, color=FRAUD_COLOR,
            label=f"Fraud (n={official['actual_fraud']:,})")
    ax.axvline(threshold, color="black", ls="--", lw=1.2, label=f"Threshold = {threshold:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction error (log scale)")
    ax.set_ylabel("Density (each class scaled to equal area)")
    ax.set_title("Baseline Autoencoder - reconstruction error on held-out test set")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ERROR_DIST_PATH, dpi=150)
    plt.close(fig)

    # 8c. Confusion matrix
    cm = np.array([[official["true_negatives"], official["false_positives"]],
                   [official["false_negatives"], official["true_positives"]]])
    row_share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({row_share[i, j]:.2%} of row)", ha="center", va="center",
                    color="white" if row_share[i, j] > 0.5 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["Predicted normal", "Predicted fraud"])
    ax.set_yticks([0, 1], ["Actual normal", "Actual fraud"])
    ax.set_title(f"Baseline confusion matrix (threshold {threshold:.4f})")
    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=150)
    plt.close(fig)

    # ==============================================================
    # 9. Limitations (plain English)
    # ==============================================================
    report.section("6. DATA LEAKAGE AND LIMITATIONS OF THE ORIGINAL EVALUATION")
    limitations = [
        f"Train/test duplicates: {removed:,} test rows have an exact copy in the training set; "
        f"{int(seen_by_autoencoder.sum()):,} normal test rows were literally seen by the Autoencoder "
        f"during training. See section 4 for their effect.",
        f"Duplicates inside the test set: {int(test_df.duplicated().sum()):,} extra copies are counted more than once.",
        "Threshold chosen on training data the model had already fitted (and the same rows fed early "
        "stopping). Training errors are optimistically low, so the threshold tends to be too low "
        "-> more false alarms on new data. No separate validation set of normal rows exists.",
        "Random (not time-based) split over 48 hours: the model learned from transactions that happen "
        "AFTER some test transactions, and 'Time' is used as an input feature.",
        f"Small number of test frauds ({official['actual_fraud']}): one fraud changes recall by about "
        f"{100 / official['actual_fraud']:.2f} percentage points - see the wide confidence intervals.",
        f"Accuracy is misleading: {pct(official['accuracy'])} looks high but is below the "
        f"{pct(official['always_normal_accuracy'])} of a model that never predicts fraud.",
        f"The 95th-percentile rule flags about 5% of normal traffic by design: "
        f"{official['false_positives']:,} false alarms, {official['false_alarms_per_fraud_caught']:.1f} "
        f"per fraud caught.",
        "Possibly undertrained: 10 epochs, validation loss still decreasing, early stopping never triggered.",
        "Not exactly reproducible: no TensorFlow random seed was set, so retraining gives different weights.",
        "Model provenance: the scaler statistics match training-normal rows exactly, but the weights file "
        "itself cannot prove which rows it was trained on.",
    ]
    for number, text in enumerate(limitations, 1):
        report.add(f"{number:>2}. {text}")

    # ==============================================================
    # 10. Integrity check and file list
    # ==============================================================
    report.section("7. INTEGRITY CHECK AND OUTPUT FILES")
    unchanged = all(sha256_of(Path(p)) == h for p, h in hashes_before.items())
    report.add(f"Protected input files unchanged after run: {unchanged}")
    for path in [REPORT_PATH, METRICS_CSV_PATH, SCORES_CSV_PATH, PR_CURVE_PATH, ERROR_DIST_PATH, CONFUSION_PATH]:
        report.add(str(path))
    report.save(REPORT_PATH)
    if not unchanged:
        raise RuntimeError("A protected input file changed during the run!")


if __name__ == "__main__":
    main()
