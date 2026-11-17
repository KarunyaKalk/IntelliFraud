"""
Phase 1 automated tests for the IntelliFraud dashboard.

Run from the project folder (virtual environment active):
    python -m pytest tests/ -v

What is protected
-----------------
* The verified baseline: the modified scoring code must reproduce the committed
  results/metrics/baseline_test_scores.csv BIT-FOR-BIT and the original
  confusion matrix TN 53,886 / FP 2,978 / FN 13 / TP 85.
* Nothing here trains a model or writes to data/, models/ or results/.

The baseline CSV is read with float_precision="round_trip" (pandas' exact
number reader). pandas' default reader can differ in the 16th-17th digit,
which is a property of CSV parsing, not of the model.

Tests that need data/creditcard.csv are skipped if the file is missing.
"""

import io
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.anomaly_detection as ad
from src.anomaly_detection import (
    InputDataError,
    ModelLoadError,
    detect_anomalies,
    load_model_components,
    select_model_features,
)
from src.dashboard_utils import (
    architecture_text,
    build_download,
    comparison_summary,
    dataset_identity,
    load_baseline_metrics,
    log_histogram,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
BASELINE_SCORES = "results/metrics/baseline_test_scores.csv"
APP_PATH = str(ROOT / "app.py")
APP_TIMEOUT = 300

# The verified baseline (never change these to make a test pass)
EXPECTED_CONFUSION = {"TN": 53_886, "FP": 2_978, "FN": 13, "TP": 85}
EXPECTED_FULL_DATASET_FLAGS = 14_769

needs_data = pytest.mark.skipif(not DATA_PATH.exists(), reason="data/creditcard.csv not present")


# ============================================================
# Shared fixtures (loaded once per test run)
# ============================================================

@pytest.fixture(scope="module")
def components():
    return load_model_components()


@pytest.fixture(scope="module")
def full_df():
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def committed_baseline():
    """The baseline scores exactly as committed in git (not the working copy)."""
    raw = subprocess.check_output(["git", "show", f"HEAD:{BASELINE_SCORES}"], cwd=ROOT)
    return pd.read_csv(io.BytesIO(raw), float_precision="round_trip")


@pytest.fixture(scope="module")
def test_rows(full_df, committed_baseline):
    return full_df.loc[committed_baseline["test_row_index"]]


def confusion(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


# ============================================================
# T1-T2: the verified baseline is unchanged
# ============================================================

@needs_data
def test_t1_baseline_reproduced_bit_for_bit(test_rows, committed_baseline, components):
    for label, results in [
        ("components passed in", detect_anomalies(test_rows, components)),
        ("components loaded by default", detect_anomalies(test_rows)),
    ]:
        assert np.array_equal(
            results["reconstruction_error"].to_numpy(),
            committed_baseline["reconstruction_error"].to_numpy(),
        ), f"reconstruction errors differ from committed baseline ({label})"
        assert np.array_equal(
            results["anomaly"].to_numpy(), committed_baseline["predicted_class"].to_numpy()
        ), f"predictions differ from committed baseline ({label})"
        assert confusion(results["Class"], results["anomaly"]) == EXPECTED_CONFUSION


@needs_data
def test_t2_full_dataset_still_flags_14769(full_df, components):
    assert int(detect_anomalies(full_df, components)["anomaly"].sum()) == EXPECTED_FULL_DATASET_FLAGS


# ============================================================
# T3-T6: input handling
# ============================================================

@needs_data
def test_t3_downloaded_file_can_be_uploaded_again(test_rows, components):
    first = detect_anomalies(test_rows, components)
    anomalies = first[first["anomaly"] == 1]
    downloaded = pd.read_csv(io.BytesIO(build_download(anomalies, components[2])))
    assert {"reconstruction_error", "anomaly", "flag_reason"} <= set(downloaded.columns)

    # 1. The 30 model inputs survive the download -> upload round trip EXACTLY.
    features = list(components[1].feature_names_in_)
    assert np.array_equal(downloaded[features].to_numpy(), anomalies[features].to_numpy())

    second = detect_anomalies(downloaded, components)  # must not raise

    # 2. Every prediction is identical.
    assert len(second) == len(anomalies)
    assert (second["anomaly"] == 1).all()

    # 3. Reconstruction errors agree within float32 precision.
    # The Autoencoder computes in 32-bit floats (relative precision ~1.2e-7).
    # Scoring 3,063 rows instead of 56,962 changes how rows are grouped into
    # batches of 512, which can change the last float32 digit (observed max
    # relative difference 7.5e-8, with identical inputs and no CSV involved).
    # The verified baseline itself is checked bit-for-bit in test_t1.
    np.testing.assert_allclose(
        second["reconstruction_error"].to_numpy(),
        anomalies["reconstruction_error"].to_numpy(),
        rtol=1e-6,
    )
    assert "flag_reason" not in second.columns  # old result columns are dropped


@needs_data
def test_t4_column_order_and_extra_columns_do_not_matter(test_rows, committed_baseline, components):
    shuffled = test_rows[list(reversed(test_rows.columns))].copy()
    shuffled["customer_note"] = "extra text column"
    shuffled["reconstruction_error"] = -1.0  # stale value from an old file
    results = detect_anomalies(shuffled, components)
    assert np.array_equal(
        results["reconstruction_error"].to_numpy(),
        committed_baseline["reconstruction_error"].to_numpy(),
    )
    _, ignored = select_model_features(shuffled, components[1])
    assert set(ignored) == {"Class", "customer_note", "reconstruction_error"}


@needs_data
@pytest.mark.parametrize("missing", [["V3"], ["Amount"], ["V3", "Amount"]])
def test_t5_missing_columns_are_named(test_rows, components, missing):
    with pytest.raises(InputDataError) as error:
        detect_anomalies(test_rows.drop(columns=missing).head(50), components)
    for column in missing:
        assert column in str(error.value)
    assert "missing" in str(error.value)


@needs_data
def test_t6_text_empty_infinite_and_empty_file(test_rows, components):
    sample = test_rows.head(20).copy()

    text = sample.copy()
    text["Amount"] = text["Amount"].astype(object)
    text.iloc[3, text.columns.get_loc("Amount")] = "abc"
    with pytest.raises(InputDataError, match="Amount"):
        detect_anomalies(text, components)

    empty_cell = sample.copy()
    empty_cell.iloc[2, empty_cell.columns.get_loc("V7")] = np.nan
    with pytest.raises(InputDataError, match="empty cells: V7"):
        detect_anomalies(empty_cell, components)

    infinite = sample.copy()
    infinite.iloc[1, infinite.columns.get_loc("V1")] = np.inf
    with pytest.raises(InputDataError, match="infinite"):
        detect_anomalies(infinite, components)

    with pytest.raises(InputDataError, match="0 rows"):
        detect_anomalies(sample.iloc[0:0], components)


# ============================================================
# T7-T8 and other helpers
# ============================================================

def test_t7_histogram_keeps_every_value():
    rng = np.random.default_rng(0)
    errors = np.concatenate([rng.lognormal(-1, 0.6, 5000), [250.0, 900.0]])  # extreme tail
    threshold = 1.0
    hist = log_histogram(errors, threshold)
    assert hist["count"].sum() == len(errors)
    assert hist["bin_end"].max() >= errors.max()
    assert hist["bin_start"].min() <= errors.min()
    flagged = hist.loc[hist["group"] == "Anomaly (flagged)", "count"].sum()
    assert flagged == int((errors > threshold).sum())


def test_t8_dataset_identity_changes_with_the_file():
    a = SimpleNamespace(file_id="abc", name="a.csv", size=10)
    b = SimpleNamespace(file_id="xyz", name="a.csv", size=10)
    assert dataset_identity(a, "unused") == dataset_identity(a, "unused")
    assert dataset_identity(a, "unused") != dataset_identity(b, "unused")
    if DATA_PATH.exists():
        assert dataset_identity(None, DATA_PATH).startswith("default:")
        assert dataset_identity(None, DATA_PATH) != dataset_identity(a, DATA_PATH)


def test_architecture_read_from_model(components):
    assert architecture_text(components[0]) == "30 → 16 → 8 → 16 → 30"


def test_baseline_metrics_file_matches_verified_values():
    metrics = load_baseline_metrics(ROOT / "results" / "metrics" / "baseline_metrics.csv")
    found = {k: int(metrics[name]["value"]) for k, name in [
        ("TN", "true_negatives"), ("FP", "false_positives"),
        ("FN", "false_negatives"), ("TP", "true_positives")]}
    assert found == EXPECTED_CONFUSION
    assert int(metrics["test_rows"]["value"]) == 56_962


def test_comparison_summary_is_honest():
    comparison = pd.read_csv(ROOT / "models" / "model_comparison.csv")
    f1 = load_baseline_metrics(ROOT / "results" / "metrics" / "baseline_metrics.csv")["f1_score"]
    text = comparison_summary(comparison, (f1["low"], f1["high"]))
    assert "Isolation Forest" in text and "1 more fraud case(s)" in text
    assert "no model is clearly better" in text


def test_model_load_errors_are_clear(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load_model_components(model_path=tmp_path / "missing.keras")
    bad_threshold = tmp_path / "threshold.txt"
    for content in ["abc", "-1", "nan"]:
        bad_threshold.write_text(content)
        with pytest.raises(ModelLoadError):
            load_model_components(threshold_path=bad_threshold)


# ============================================================
# T9-T15: the dashboard itself (Streamlit AppTest, headless)
# ============================================================

def page_text(at):
    parts = []
    for group in ["title", "header", "subheader", "markdown", "caption", "info",
                  "warning", "error", "success", "text"]:
        parts += [str(element.value) for element in getattr(at, group)]
    parts += [str(tab.label) for tab in at.tabs]
    parts += [f"{m.label}: {m.value}" for m in at.metric]
    parts += [str(b.label) for b in at.button]
    return "\n".join(parts)


def metric_value(at, label):
    for m in at.metric:
        if m.label == label:
            return m.value
    return None


@pytest.fixture
def fresh_cache():
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def upload(at, name, frame):
    at.file_uploader[0].upload(name, frame.to_csv(index=False).encode("utf-8"), "text/csv")
    return at.run()


@needs_data
def test_t9_t10_default_dataset_labels_and_detection():
    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()
    assert not at.exception
    text = page_text(at)
    assert "Fraud Transactions" not in text
    assert "Anomalies Detected" in text
    assert "56,962 held-out test transactions" in text
    assert "do **not** change when you upload" in text
    assert "not** the same as confirmed fraud" in text
    # Training-split warning on the default dataset: 227,845 rows in the training
    # split, of which only the 227,451 legitimate ones trained the Autoencoder
    # and set the threshold; the 394 fraudulent ones were excluded.
    assert "227,845 of these rows (about 80%) belong to the training split" in text
    assert "227,451 legitimate transactions were used to train the Autoencoder and set its threshold" in text
    assert "394 fraudulent training-split transactions were excluded from training" in text
    assert "227,845) were used to train" not in text  # the old, inaccurate wording

    at.button(key="detect_button").click().run()
    assert not at.exception
    assert metric_value(at, "Anomalies Detected") == "14,769"
    assert metric_value(at, "Transactions Scored") == "284,807"


@needs_data
def test_t11_stale_results_are_not_shown(full_df):
    sample_a = full_df.sample(2000, random_state=1)
    sample_b = full_df.sample(3000, random_state=2)

    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()
    upload(at, "a.csv", sample_a)
    at.button(key="detect_button").click().run()
    assert metric_value(at, "Transactions Scored") == "2,000"

    upload(at, "b.csv", sample_b)  # new file, no click yet
    assert not at.exception
    assert metric_value(at, "Transactions Scored") is None
    assert "previous results were cleared" in page_text(at)

    at.button(key="detect_button").click().run()
    assert metric_value(at, "Transactions Scored") == "3,000"


@needs_data
def test_t12_dashboard_download_can_be_uploaded(full_df, components):
    sample = full_df.sample(3000, random_state=3)
    results = detect_anomalies(sample, components)
    downloaded = pd.read_csv(io.BytesIO(build_download(results[results.anomaly == 1], components[2])))

    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()
    upload(at, "intellifraud_anomalies.csv", downloaded)
    at.button(key="detect_button").click().run()
    assert not at.exception
    assert not [e.value for e in at.error]
    assert metric_value(at, "Anomalies Detected") == f"{len(downloaded):,}"
    assert "were ignored by the model" in page_text(at)


@needs_data
def test_t12b_bad_upload_shows_understandable_error(full_df):
    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()
    upload(at, "bad.csv", full_df.head(100).drop(columns=["V3"]))
    at.button(key="detect_button").click().run()
    assert not at.exception
    errors = " ".join(str(e.value) for e in at.error)
    assert "cannot be scored" in errors and "V3" in errors
    assert metric_value(at, "Transactions Scored") is None


def test_t13_model_load_failure_disables_detection(monkeypatch, fresh_cache, tmp_path):
    monkeypatch.setattr(ad, "MODEL_PATH", tmp_path / "does_not_exist.keras")  # in memory only
    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()
    assert not at.exception
    errors = " ".join(str(e.value) for e in at.error)
    assert "could not be loaded" in errors and "does_not_exist.keras" in errors
    if DATA_PATH.exists():
        assert at.button(key="detect_button").disabled
    assert "Model not loaded" in page_text(at)


def test_t14_no_deprecated_width_arguments():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "use_container_width" not in source


@needs_data
def test_t14b_no_deprecation_warnings_on_page():
    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()
    at.button(key="detect_button").click().run()
    assert not at.exception
    assert not [w.value for w in at.warning if "deprecat" in str(w.value).lower()]


@needs_data
def test_t15_model_loaded_once(monkeypatch, fresh_cache):
    calls = {"count": 0}
    original = ad.load_model_components

    def counting_loader(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ad, "load_model_components", counting_loader)
    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()
    at.button(key="detect_button").click().run()
    at.button(key="detect_button").click().run()
    assert not at.exception
    assert calls["count"] == 1
