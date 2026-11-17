"""
Helper functions for the IntelliFraud Streamlit dashboard (app.py).

These are plain Python/pandas functions with no Streamlit code inside,
so they can be tested on their own (see tests/test_phase1.py).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.anomaly_detection as anomaly_detection

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# AVAILABLE MODELS AND DATASETS
# ============================================================
# Each model is a fixed set of THREE files (model, scaler, threshold) that
# always belong together. The dashboard never mixes files between models.

MODEL_OPTIONS = {
    "baseline": {
        "label": "Baseline Autoencoder",
        "tag": "BASELINE",
        "summary": "Original model: trained on 227,451 legitimate training transactions; "
                   "threshold = 95th percentile of its training errors.",
    },
    "improved": {
        "label": "Improved Autoencoder (v2)",
        "tag": "IMPROVED-V2",
        "summary": "Same architecture and features; de-duplicated training data, separate "
                   "validation set, threshold = 95th percentile of validation errors.",
    },
}

DATASET_OPTIONS = {
    "original": {
        "label": "Original Credit Card Dataset",
        "tag": "ORIGINAL",
        "path": PROJECT_ROOT / "data" / "creditcard.csv",
        "synthetic": False,
        "how_to_get": "Download the Kaggle 'Credit Card Fraud Detection' dataset (mlg-ulb/creditcardfraud) "
                      "and save it as data/creditcard.csv.",
    },
    "synthetic": {
        "label": "Synthetic Credit Card Dataset (5% fraud)",
        "tag": "SYNTHETIC-5PCT",
        "path": PROJECT_ROOT / "data" / "synthetic" / "creditcard_synthetic_5pct.csv",
        "metadata": PROJECT_ROOT / "data" / "synthetic" / "synthetic_data_metadata.json",
        "synthetic": True,
        "how_to_get": "Recreate it with:  python src/generate_synthetic_data.py   "
                      "(needs data/creditcard.csv; seed 42 gives the identical file).",
    },
}

IMPROVED_DIR = PROJECT_ROOT / "models" / "improved"


def model_file_paths(model_key):
    """
    (model, scaler, threshold) paths for one model - always from the same folder.

    The baseline paths are read from src/anomaly_detection at call time, so the
    dashboard and the original scripts always agree on them.
    """
    if model_key == "baseline":
        return anomaly_detection.MODEL_PATH, anomaly_detection.SCALER_PATH, anomaly_detection.THRESHOLD_PATH
    if model_key == "improved":
        return IMPROVED_DIR / "autoencoder.keras", IMPROVED_DIR / "scaler.pkl", IMPROVED_DIR / "threshold.txt"
    raise ValueError(f"Unknown model: {model_key}")


def improved_training_counts(config_path=None):
    """Row counts recorded when Improved v2 was trained (None if unavailable)."""
    path = Path(config_path or IMPROVED_DIR / "config.json")
    if not path.exists():
        return None
    return json.loads(path.read_text())["row_counts"]


def read_synthetic_metadata(metadata_path=None):
    path = Path(metadata_path or DATASET_OPTIONS["synthetic"]["metadata"])
    if not path.exists():
        return None
    return json.loads(path.read_text())


def synthetic_start_row(metadata):
    """Row position where generated rows begin (all rows before it are original)."""
    if not metadata:
        return None
    return int(metadata["synthetic_rows"]["first_row_position"])


# ============================================================
# DATASET OVERVIEW
# ============================================================

def has_usable_labels(df):
    return "Class" in df.columns and pd.api.types.is_numeric_dtype(df["Class"]) \
        and set(df["Class"].dropna().unique()) <= {0, 1}


def dataset_summary(df):
    """Counts for the metric cards. Fraud figures are None if Class is missing."""
    summary = {
        "total": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "has_labels": has_usable_labels(df),
        "legitimate": None,
        "fraud": None,
        "fraud_percentage": None,
    }
    if summary["has_labels"] and len(df):
        summary["fraud"] = int((df["Class"] == 1).sum())
        summary["legitimate"] = int((df["Class"] == 0).sum())
        summary["fraud_percentage"] = 100 * summary["fraud"] / len(df)
    return summary


def amount_histogram(df, synthetic_start=None, bins=40):
    """
    Share of transactions per log10(Amount + 1) bin, for each group:
    'Legitimate', 'Fraud (original)' and - for the synthetic dataset -
    'Fraud (synthetic)'. Without labels everything is 'All transactions'.
    Each group sums to 1, so groups of very different size can be compared.
    """
    if "Amount" not in df.columns or not pd.api.types.is_numeric_dtype(df["Amount"]):
        return pd.DataFrame(columns=["bin_start", "bin_end", "group", "share"])

    amounts = np.log10(df["Amount"].clip(lower=0) + 1)
    edges = np.linspace(0, max(float(amounts.max()), 1e-9), bins + 1)

    if has_usable_labels(df):
        position = np.arange(len(df))
        is_fraud = df["Class"].to_numpy() == 1
        is_synthetic = position >= synthetic_start if synthetic_start is not None else np.zeros(len(df), bool)
        groups = [("Legitimate", ~is_fraud), ("Fraud (original)", is_fraud & ~is_synthetic)]
        if is_synthetic.any():
            groups.append(("Fraud (synthetic)", is_fraud & is_synthetic))
    else:
        groups = [("All transactions", np.ones(len(df), bool))]

    rows = []
    for name, mask in groups:
        if not mask.any():
            continue
        counts, _ = np.histogram(amounts[mask], bins=edges)
        for start, end, count in zip(edges[:-1], edges[1:], counts):
            rows.append({"bin_start": start, "bin_end": end, "group": name, "share": count / mask.sum()})
    return pd.DataFrame(rows, columns=["bin_start", "bin_end", "group", "share"])


def class_groups(df, synthetic_start=None):
    """
    Boolean masks for each class group in a labelled file:
    'Legitimate', 'Fraud' - or, when the file contains generated rows,
    'Fraud (original)' and 'Fraud (synthetic)'. Returns [] without labels.
    """
    if not has_usable_labels(df):
        return []
    is_fraud = df["Class"].to_numpy() == 1
    if synthetic_start is None:
        return [("Legitimate", ~is_fraud), ("Fraud", is_fraud)]
    is_synthetic = np.arange(len(df)) >= synthetic_start
    return [("Legitimate", ~is_fraud), ("Fraud (original)", is_fraud & ~is_synthetic),
            ("Fraud (synthetic)", is_fraud & is_synthetic)]


def class_distribution(df, synthetic_start=None):
    """Transactions and share of the file for each class group (empty without labels)."""
    rows = [{"group": name, "transactions": int(mask.sum()),
             "share": float(mask.sum() / len(df)) if len(df) else 0.0}
            for name, mask in class_groups(df, synthetic_start)]
    return pd.DataFrame(rows, columns=["group", "transactions", "share"])


def amount_statistics(df, synthetic_start=None):
    """Amount summary per class group, computed from the loaded file."""
    columns = ["Group", "Transactions", "Median amount", "Mean amount", "95th percentile", "Largest amount"]
    if "Amount" not in df.columns or not pd.api.types.is_numeric_dtype(df["Amount"]):
        return pd.DataFrame(columns=columns)
    groups = class_groups(df, synthetic_start) or [("All transactions", np.ones(len(df), bool))]
    rows = []
    for name, mask in groups:
        amounts = df["Amount"].to_numpy()[mask]
        amounts = amounts[~np.isnan(amounts)]
        if len(amounts) == 0:
            continue
        rows.append([name, int(mask.sum()), float(np.median(amounts)), float(amounts.mean()),
                     float(np.percentile(amounts, 95)), float(amounts.max())])
    return pd.DataFrame(rows, columns=columns)


# ============================================================
# ANOMALY SCORE (error / threshold) BANDS
# ============================================================
# The anomaly score is the reconstruction error divided by the model's
# threshold: above 1 means flagged. Bands make the shape readable.

SCORE_BAND_EDGES = [0.5, 1.0, 2.0, 5.0, 10.0]
SCORE_BAND_LABELS = ["below 0.5x", "0.5x - 1x", "1x - 2x", "2x - 5x", "5x - 10x", "10x and above"]


def anomaly_score_bands(errors, threshold):
    """
    Count transactions per anomaly-score band. A score of exactly 1 is NOT
    flagged (flagging uses error > threshold), so it falls in '0.5x - 1x'.
    """
    scores = np.asarray(errors, dtype=float) / float(threshold)
    band = np.digitize(scores, SCORE_BAND_EDGES, right=True)
    counts = np.bincount(band, minlength=len(SCORE_BAND_LABELS))
    return pd.DataFrame({
        "band": SCORE_BAND_LABELS,
        "order": range(len(SCORE_BAND_LABELS)),
        "transactions": counts.astype(int),
        "share": counts / max(len(scores), 1),
        "result": ["Not flagged", "Not flagged"] + ["Anomaly (flagged)"] * 4,
    })


# ============================================================
# STORED TEST-SET SCORES -> LIVE REPORT CHARTS (no metric recalculated)
# ============================================================
# results/metrics/*_test_scores.csv hold every official test row's stored
# reconstruction error and label. The functions below only reshape them for
# plotting; the reported metric values are always read from the metric files.

def pr_curve_points(labels, scores, max_points=1500):
    """
    Precision-recall curve from stored scores, thinned for drawing.
    Every point where recall changes is kept, so the shape is exact.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(np.asarray(labels), np.asarray(scores))
    change = np.flatnonzero(np.diff(recall) != 0)
    keep = np.union1d(np.union1d(change, change + 1), np.linspace(0, len(recall) - 1, max_points).astype(int))
    return pd.DataFrame({"recall": recall[keep], "precision": precision[keep]})


def class_error_histogram(errors, labels, edges):
    """Share of each class (legitimate / fraud) per error bin, on shared bin edges."""
    errors = np.clip(np.asarray(errors, dtype=float), edges[0], edges[-1])
    labels = np.asarray(labels)
    rows = []
    for name, mask in [("Legitimate", labels == 0), ("Fraud", labels == 1)]:
        if not mask.any():
            continue
        counts, _ = np.histogram(errors[mask], bins=edges)
        for start, end, count in zip(edges[:-1], edges[1:], counts):
            rows.append({"bin_start": start, "bin_end": end, "group": name, "count": int(count),
                         "share": count / mask.sum()})
    return pd.DataFrame(rows, columns=["bin_start", "bin_end", "group", "count", "share"])


def shared_log_edges(error_arrays, bins=50):
    """Log-spaced bin edges covering every array, so models share one x-axis."""
    values = np.concatenate([np.asarray(a, dtype=float) for a in error_arrays])
    positive = values[values > 0]
    return np.geomspace(positive.min(), values.max() * 1.0001, bins + 1)


def confusion_frame(official_table):
    """Long-format confusion matrices for both models, read from the official table."""
    by_key = official_table.set_index("key")
    cells = [("Legitimate", "Not flagged", "true_negatives", "True negative"),
             ("Legitimate", "Flagged", "false_positives", "False positive"),
             ("Fraud", "Not flagged", "false_negatives", "False negative"),
             ("Fraud", "Flagged", "true_positives", "True positive")]
    rows = []
    for model, column in [("Baseline", "baseline"), ("Improved v2", "improved")]:
        for actual, predicted, key, outcome in cells:
            rows.append({"model": model, "actual": actual, "predicted": predicted, "outcome": outcome,
                         "correct": outcome.startswith("True"),
                         "count": int(round(by_key.loc[key, column]))})
    return pd.DataFrame(rows)


def training_history(path, model_name):
    """Training and validation loss per epoch, long format (epoch numbers start at 1)."""
    history = pd.read_csv(path)
    if "epoch" not in history.columns:
        history.insert(0, "epoch", np.arange(1, len(history) + 1))
    long = history.melt(id_vars="epoch", value_vars=["loss", "val_loss"], var_name="split", value_name="loss_value")
    long["split"] = long["split"].map({"loss": "Training loss", "val_loss": "Validation loss"})
    long["model"] = model_name
    return long


CI_CHART_METRICS = ["precision", "recall", "f1_score", "average_precision", "roc_auc"]
PAIRED_CHART_METRICS = ["average_precision", "pr_auc_trapezoidal", "roc_auc", "recall", "precision", "f1_score"]


def ci_frame(official_table, keys=CI_CHART_METRICS):
    """Value and 95% CI per model and metric, long format, from the official table."""
    rows = []
    for _, row in official_table[official_table["key"].isin(keys)].iterrows():
        for model, prefix in [("Baseline", "baseline"), ("Improved v2", "improved")]:
            rows.append({"metric": row["metric"], "key": row["key"], "model": model, "value": row[prefix],
                         "low": row[f"{prefix}_low"], "high": row[f"{prefix}_high"]})
    return pd.DataFrame(rows)


def paired_difference_frame(official_table, keys=PAIRED_CHART_METRICS):
    """Observed difference (Improved v2 - baseline) and its paired 95% CI, from the official table."""
    table = official_table.set_index("key")
    rows = [{"metric": table.loc[k, "metric"], "key": k, "difference": table.loc[k, "difference"],
             "low": table.loc[k, "paired_low"], "high": table.loc[k, "paired_high"],
             "primary": k == "average_precision"} for k in keys if k in table.index]
    return pd.DataFrame(rows)


# ============================================================
# PREDICTED ANOMALY vs KNOWN LABEL (counts only - not an evaluation)
# ============================================================

def label_comparison(results):
    """
    2 x 2 table of counts: predicted anomaly (rows) vs known Class label
    (columns). Returns None if the data has no usable Class column.
    Deliberately counts only - no precision/recall - because on these files
    the comparison is not a fair evaluation (see the dashboard warnings).
    """
    if not has_usable_labels(results):
        return None
    predicted = results["anomaly"].to_numpy() == 1
    labelled_fraud = results["Class"].to_numpy() == 1
    table = pd.DataFrame(
        {
            "Labelled fraud (Class = 1)": [int((predicted & labelled_fraud).sum()),
                                           int((~predicted & labelled_fraud).sum())],
            "Labelled legitimate (Class = 0)": [int((predicted & ~labelled_fraud).sum()),
                                                int((~predicted & ~labelled_fraud).sum())],
        },
        index=["Predicted anomaly", "Not flagged"],
    )
    table["Total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum(axis=0)
    return table


# ============================================================
# COMPACT FULL-RESULTS DOWNLOAD
# ============================================================

def compact_results_csv(results, model_label, threshold):
    """
    One row per scored transaction: row number, known label (if any),
    reconstruction error and prediction. About 9 MB for 300,000 rows,
    instead of ~160 MB for every original column.
    """
    export = pd.DataFrame({"row_number": np.arange(len(results))})
    if has_usable_labels(results):
        export["known_label_class"] = results["Class"].to_numpy().astype(int)
    export["reconstruction_error"] = results["reconstruction_error"].to_numpy()
    export["predicted_anomaly"] = results["anomaly"].to_numpy().astype(int)
    export["threshold"] = threshold
    export["model"] = model_label
    return export.to_csv(index=False).encode("utf-8")


# ============================================================
# OFFICIAL RESULTS: BASELINE vs IMPROVED v2 (read only)
# ============================================================

COMPARISON_METRICS = [
    ("Precision", "precision", "pct"),
    ("Recall", "recall", "pct"),
    ("F1-score", "f1_score", "pct"),
    ("Average Precision", "average_precision", "num"),
    ("PR-AUC (trapezoidal)", "pr_auc_trapezoidal", "num"),
    ("ROC-AUC", "roc_auc", "num"),
    ("False-positive rate", "false_positive_rate", "pct"),
    ("Specificity", "specificity", "pct"),
    ("False alarms per fraud caught", "false_alarms_per_fraud_caught", "ratio"),
    ("True positives (fraud caught)", "true_positives", "int"),
    ("False negatives (fraud missed)", "false_negatives", "int"),
    ("False positives (false alarms)", "false_positives", "int"),
    ("True negatives", "true_negatives", "int"),
]


def format_metric(value, kind):
    if value is None or pd.isna(value):
        return "-"
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "num":
        return f"{value:.4f}"
    if kind == "ratio":
        return f"{value:.1f}"
    return f"{int(round(value)):,}"


def load_official_comparison(baseline_metrics_path, improved_metrics_path, comparison_path):
    """
    Combine the committed, verified result files into one table.
    Nothing is recalculated. Returns (table, conclusion).
    """
    def official(path):
        frame = pd.read_csv(path)
        return frame[frame["evaluation_set"] == "official_full_test"].set_index("metric")

    baseline = official(baseline_metrics_path)
    improved = official(improved_metrics_path)
    paired = pd.read_csv(comparison_path).set_index("metric")

    rows = []
    for label, key, kind in COMPARISON_METRICS:
        rows.append({
            "metric": label, "key": key, "kind": kind,
            "baseline": baseline.loc[key, "value"],
            "baseline_low": baseline.loc[key, "ci95_low"], "baseline_high": baseline.loc[key, "ci95_high"],
            "improved": improved.loc[key, "value"],
            "improved_low": improved.loc[key, "ci95_low"], "improved_high": improved.loc[key, "ci95_high"],
            "difference": improved.loc[key, "value"] - baseline.loc[key, "value"],
            "paired_low": paired.loc[key, "paired_ci95_low"] if key in paired.index else np.nan,
            "paired_high": paired.loc[key, "paired_ci95_high"] if key in paired.index else np.nan,
        })
    table = pd.DataFrame(rows)

    ap = paired.loc["average_precision"]
    conclusion = {
        "supported": bool(paired.loc["improvement_statistically_supported", "difference_improved_minus_baseline"]),
        "ap_baseline": float(ap["baseline"]),
        "ap_improved": float(ap["improved_v2"]),
        "ap_difference": float(ap["difference_improved_minus_baseline"]),
        "ap_low": float(ap["paired_ci95_low"]),
        "ap_high": float(ap["paired_ci95_high"]),
        "test_rows": int(baseline.loc["test_rows", "value"]),
        "test_fraud": int(baseline.loc["actual_fraud", "value"]),
        "baseline_threshold": float(baseline.loc["threshold", "value"]),
        "improved_threshold": float(improved.loc["threshold", "value"]),
    }
    return table, conclusion


# ============================================================
# TRANSACTION CASE FILES
# ============================================================
# A case file is a readable summary of ONE flagged transaction, built only
# from the current detection results. The reference is derived from the row
# number in the loaded file, so it is reproducible and traceable - it is not
# a stored database record.

INVESTIGATION_NOTE = ("The reconstruction error exceeds the configured anomaly threshold. "
                      "Further review is recommended.")


def case_reference(row_position):
    """Display reference for the row at this position in the loaded file, e.g. IF-000421."""
    return f"IF-{int(row_position):06d}"


def describe_time(seconds):
    """'Time' is seconds since the first transaction in the file."""
    if seconds is None or pd.isna(seconds):
        return "Not available"
    seconds = float(seconds)
    day, rest = divmod(int(seconds), 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes, secs = divmod(rest, 60)
    return f"{seconds:,.0f} s after the first transaction (day {day + 1}, {hours:02d}:{minutes:02d}:{secs:02d})"


def flagged_positions(results):
    """Row positions of flagged transactions, highest reconstruction error first."""
    errors = results["reconstruction_error"].to_numpy()
    flagged = np.flatnonzero(results["anomaly"].to_numpy() == 1)
    return flagged[np.argsort(-errors[flagged], kind="stable")]


def build_case_file(results, row_position, threshold, model_label, synthetic_start=None):
    """
    Facts for one case file, taken directly from the detection results.
    No values are invented; anything missing is reported as 'Not available'.
    """
    row = results.iloc[int(row_position)]
    error = float(row["reconstruction_error"])
    flagged = int(row["anomaly"]) == 1

    if "Class" in results.columns and has_usable_labels(results):
        known_label = "Fraud (Class = 1)" if int(row["Class"]) == 1 else "Legitimate (Class = 0)"
    else:
        known_label = "Not available"

    amount = row["Amount"] if "Amount" in results.columns else None
    origin = None
    if synthetic_start is not None:
        origin = "Synthetic (generated)" if row_position >= synthetic_start else "Original dataset row"

    return {
        "reference": case_reference(row_position),
        "row_number": int(row_position),
        "status": "ANOMALY FLAGGED" if flagged else "NOT FLAGGED",
        "classification": "Anomaly" if flagged else "Normal",
        "amount": None if amount is None or pd.isna(amount) else float(amount),
        "time": describe_time(row["Time"]) if "Time" in results.columns else "Not available",
        "reconstruction_error": error,
        "threshold": float(threshold),
        "error_to_threshold_ratio": error / float(threshold),
        "known_label": known_label,
        "row_origin": origin,
        "source_model": model_label,
        "note": INVESTIGATION_NOTE if flagged else "The reconstruction error is within the threshold.",
    }


def case_context(results, row_position):
    """
    Where this transaction sits in the current results: its rank among the
    flagged transactions (1 = highest error) and the share of all scored
    transactions with a lower reconstruction error.
    """
    errors = results["reconstruction_error"].to_numpy()
    error = errors[int(row_position)]
    ranked = flagged_positions(results)
    matches = np.flatnonzero(ranked == int(row_position))
    return {
        "flagged_rank": int(matches[0]) + 1 if len(matches) else None,
        "flagged_total": int(len(ranked)),
        "share_below": float((errors < error).mean()),
    }


# ============================================================
# WHICH DATASET IS LOADED?
# ============================================================

def dataset_identity(uploaded_file, default_path):
    """
    Return a short text that identifies the dataset currently loaded.

    Detection results are stored together with this identity. If the user
    uploads a different file (or removes the upload), the identity changes
    and the old results are no longer shown.
    """
    if uploaded_file is not None:
        file_id = getattr(uploaded_file, "file_id", None)
        if file_id:
            return f"upload:{file_id}"
        return f"upload:{uploaded_file.name}:{uploaded_file.size}"

    path = Path(default_path)
    stat = path.stat()
    return f"default:{path}:{stat.st_size}:{stat.st_mtime_ns}"


# ============================================================
# MODEL DESCRIPTION
# ============================================================

def architecture_text(model):
    """Read the layer sizes from the loaded model, e.g. '30 → 16 → 8 → 16 → 30'."""
    sizes = [int(model.input_shape[-1])]
    sizes += [int(layer.units) for layer in model.layers if hasattr(layer, "units")]
    return " → ".join(str(size) for size in sizes)


# ============================================================
# RECONSTRUCTION-ERROR HISTOGRAM
# ============================================================

def log_histogram(errors, threshold, bins=60):
    """
    Histogram data for ALL reconstruction errors (nothing is clipped).

    Bins are spaced evenly on a log scale from the smallest to the largest
    error. Each bin is split into 'Not flagged' (error <= threshold) and
    'Anomaly (flagged)' (error > threshold), so the bars can be coloured.

    Returns a DataFrame with columns bin_start, bin_end, group, count.
    """
    errors = np.asarray(errors, dtype=float)
    positive = errors[errors > 0]
    low = positive.min() if len(positive) else 1e-12
    high = max(errors.max(), low * 1.0001)
    edges = np.geomspace(low, high, bins + 1)
    clipped = np.clip(errors, low, high)  # only affects errors of exactly 0

    rows = []
    for group, mask in [("Not flagged", errors <= threshold), ("Anomaly (flagged)", errors > threshold)]:
        counts, _ = np.histogram(clipped[mask], bins=edges)
        for start, end, count in zip(edges[:-1], edges[1:], counts):
            if count:
                rows.append({"bin_start": start, "bin_end": end, "group": group, "count": int(count)})

    return pd.DataFrame(rows, columns=["bin_start", "bin_end", "group", "count"])


# ============================================================
# DOWNLOADABLE RESULTS
# ============================================================

def build_download(results, threshold):
    """
    CSV bytes for download: all original columns, plus reconstruction_error,
    anomaly and a plain-English flag_reason.

    The file can be uploaded to the dashboard again: the model only reads its
    own 30 input columns and recalculates the result columns.
    """
    export = results.copy()
    export = export.drop(columns=[c for c in ["flag_reason", "Why Flagged"] if c in export.columns])
    export["flag_reason"] = np.where(
        export["anomaly"] == 1,
        "Reconstruction error " + export["reconstruction_error"].map(lambda x: f"{x:.4f}")
        + f" is above the threshold {threshold:.4f}",
        "",
    )
    return export.to_csv(index=False).encode("utf-8")


# ============================================================
# STORED BASELINE AND COMPARISON RESULTS
# ============================================================

def load_baseline_metrics(path):
    """
    Read the verified baseline (official full test set) from
    results/metrics/baseline_metrics.csv.

    Returns {metric: {"value": v, "low": ci_low, "high": ci_high}}.
    """
    table = pd.read_csv(path)
    table = table[table["evaluation_set"] == "official_full_test"]
    return {
        row.metric: {"value": row.value, "low": row.ci95_low, "high": row.ci95_high}
        for row in table.itertuples()
    }


def comparison_summary(comparison_df, autoencoder_f1_ci=None):
    """
    An honest one-paragraph summary of the stored model comparison.

    Instead of naming a 'best' model from a tiny F1 difference, it states the
    difference in percentage points and in fraud cases caught, and whether it
    lies inside the Autoencoder's own 95% confidence interval for F1.
    """
    table = comparison_df.set_index("Model")
    best_name = table["F1 Score"].idxmax()
    best = table.loc[best_name]

    if "Autoencoder" not in table.index:
        return f"Highest F1 in the stored comparison: {best_name} ({best['F1 Score']:.2%})."

    autoencoder = table.loc["Autoencoder"]
    if best_name == "Autoencoder":
        return f"The Autoencoder has the highest F1 in the stored comparison ({best['F1 Score']:.2%})."

    gap_points = (best["F1 Score"] - autoencoder["F1 Score"]) * 100
    extra_cases = int(best["Fraud Caught"] - autoencoder["Fraud Caught"])
    text = (
        f"Highest F1: {best_name} ({best['F1 Score']:.2%}) vs Autoencoder "
        f"({autoencoder['F1 Score']:.2%}) - a difference of {gap_points:.2f} percentage points, "
        f"and {abs(extra_cases)} {'more' if extra_cases >= 0 else 'fewer'} fraud case(s) caught out of "
        f"{int(best['Total Fraud'])}."
    )
    if autoencoder_f1_ci is not None:
        low, high = autoencoder_f1_ci
        if low <= best["F1 Score"] <= high:
            text += (
                f" This lies inside the Autoencoder's own 95% confidence interval for F1 "
                f"({low:.2%} - {high:.2%}), so no model is clearly better on this test set."
            )
        else:
            text += (
                f" This lies outside the Autoencoder's 95% confidence interval for F1 "
                f"({low:.2%} - {high:.2%})."
            )
    return text
