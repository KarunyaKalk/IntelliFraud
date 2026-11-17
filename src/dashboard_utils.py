"""
Helper functions for the IntelliFraud Streamlit dashboard (app.py).

These are plain Python/pandas functions with no Streamlit code inside,
so they can be tested on their own (see tests/test_phase1.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd


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
