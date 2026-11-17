"""
Tests for THE FRAUD BUREAU dashboard (app.py) and its helpers (src/dashboard_utils.py).

Run from the project folder:
    python -m pytest tests/ -v

These tests never train a model and never write to data/, models/ or results/.
Tests that need a dataset are skipped if that dataset file is missing.
"""

import io
import re
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.dashboard_utils as du
from src.anomaly_detection import InputDataError, detect_anomalies, load_model_components, select_model_features
from src.dashboard_utils import (
    DATASET_OPTIONS,
    INVESTIGATION_NOTE,
    SCORE_BAND_LABELS,
    amount_histogram,
    amount_statistics,
    anomaly_score_bands,
    build_case_file,
    case_context,
    case_reference,
    ci_frame,
    class_distribution,
    class_error_histogram,
    compact_results_csv,
    confusion_frame,
    dataset_summary,
    describe_time,
    flagged_positions,
    label_comparison,
    load_official_comparison,
    model_file_paths,
    paired_difference_frame,
    pr_curve_points,
    shared_log_edges,
    training_history,
)

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "app.py")
APP_TIMEOUT = 300
ORIGINAL = DATASET_OPTIONS["original"]["path"]
SYNTHETIC = DATASET_OPTIONS["synthetic"]["path"]
METRICS = ROOT / "results" / "metrics"

needs_original = pytest.mark.skipif(not ORIGINAL.exists(), reason="data/creditcard.csv not present")
needs_synthetic = pytest.mark.skipif(not SYNTHETIC.exists(), reason="synthetic dataset not present")

LABEL_WARNING = ("These counts compare predictions with the dataset's known labels. They are not independent "
                 "model evaluation results because the selected dataset may contain transactions used during "
                 "training.")
CONCLUSION = "An improvement in the Improved v2 model has not been statistically established."


# ------------------------------------------------------------------
# Fixtures and helpers
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def models():
    return {key: load_model_components(*model_file_paths(key)) for key in ["baseline", "improved"]}


@pytest.fixture(scope="module")
def original_df():
    return pd.read_csv(ORIGINAL)


@pytest.fixture(scope="module")
def synthetic_df():
    return pd.read_csv(SYNTHETIC)


@pytest.fixture
def fresh_cache():
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def page_text(at):
    parts = []
    for group in ["title", "header", "subheader", "markdown", "caption", "info", "warning", "error",
                  "success", "text"]:
        parts += [str(element.value) for element in getattr(at, group)]
    parts += [str(tab.label) for tab in at.tabs]
    parts += [f"{m.label}: {m.value}" for m in at.metric]
    return "\n".join(parts)


def metric_value(at, label):
    for m in at.metric:
        if m.label == label:
            return m.value
    return None


def run_app():
    return AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT).run()


def git_head_bytes(path):
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)


# ------------------------------------------------------------------
# Models: each uses its own matching model, scaler and threshold
# ------------------------------------------------------------------

def test_each_model_uses_files_from_its_own_folder():
    baseline = [Path(p) for p in model_file_paths("baseline")]
    improved = [Path(p) for p in model_file_paths("improved")]
    assert {p.parent for p in baseline} == {ROOT / "models"}
    assert {p.parent for p in improved} == {ROOT / "models" / "improved"}
    assert [p.name for p in baseline] == [p.name for p in improved] == ["autoencoder.keras", "scaler.pkl",
                                                                        "threshold.txt"]
    with pytest.raises(ValueError):
        model_file_paths("unknown")


@pytest.mark.parametrize("key", ["baseline", "improved"])
def test_loaded_scaler_and_threshold_match_that_models_files(models, key):
    model, scaler, threshold = models[key]
    _, scaler_path, threshold_path = model_file_paths(key)
    assert threshold == float(Path(threshold_path).read_text())
    np.testing.assert_array_equal(scaler.mean_, joblib.load(scaler_path).mean_)
    assert model.input_shape == (None, 30)


def test_the_two_models_are_really_different(models):
    assert models["baseline"][2] != models["improved"][2]
    assert not np.array_equal(models["baseline"][1].mean_, models["improved"][1].mean_)


@pytest.mark.parametrize("key", ["baseline", "improved"])
def test_class_is_never_a_model_input(models, key):
    scaler = models[key][1]
    assert "Class" not in scaler.feature_names_in_
    frame = pd.DataFrame(np.zeros((3, 31)), columns=["Class"] + list(scaler.feature_names_in_))
    features, ignored = select_model_features(frame, scaler)
    assert list(features.columns) == list(scaler.feature_names_in_)
    assert "Class" in ignored


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------

@needs_original
def test_original_dataset_summary(original_df):
    summary = dataset_summary(original_df)
    assert summary["total"] == 284_807 and summary["fraud"] == 492 and summary["legitimate"] == 284_315
    assert abs(summary["fraud_percentage"] - 0.1727) < 1e-3
    assert summary["missing_values"] == 0


@needs_synthetic
def test_synthetic_dataset_summary_and_histogram(synthetic_df):
    summary = dataset_summary(synthetic_df)
    assert summary["total"] == 299_279 and summary["fraud"] == 14_964
    assert abs(summary["fraud_percentage"] - 5.0) < 0.01
    start = du.synthetic_start_row(du.read_synthetic_metadata())
    assert start == 284_807
    hist = amount_histogram(synthetic_df, start)
    assert set(hist["group"]) == {"Legitimate", "Fraud (original)", "Fraud (synthetic)"}
    assert np.allclose(hist.groupby("group")["share"].sum(), 1.0)


def test_summary_without_labels():
    frame = pd.DataFrame({"Amount": [1.0, 2.0]})
    summary = dataset_summary(frame)
    assert summary["has_labels"] is False and summary["fraud"] is None
    assert set(amount_histogram(frame)["group"]) == {"All transactions"}


# ------------------------------------------------------------------
# Detection results helpers: label counts, downloads, case files
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_results(original_df, models):
    sample = original_df.sample(4000, random_state=11).reset_index(drop=True)
    return detect_anomalies(sample, models["baseline"])


@needs_original
def test_label_comparison_counts(sample_results):
    table = label_comparison(sample_results)
    assert table.loc["Total", "Total"] == len(sample_results)
    assert table.loc["Predicted anomaly", "Total"] == int(sample_results["anomaly"].sum())
    assert table.loc["Total", "Labelled fraud (Class = 1)"] == int(sample_results["Class"].sum())
    assert label_comparison(sample_results.drop(columns=["Class"])) is None


@needs_original
def test_compact_download_contents(sample_results, models):
    data = pd.read_csv(io.BytesIO(compact_results_csv(sample_results, "Baseline Autoencoder", models["baseline"][2])))
    assert list(data.columns) == ["row_number", "known_label_class", "reconstruction_error", "predicted_anomaly",
                                  "threshold", "model"]
    assert len(data) == len(sample_results)
    assert data["predicted_anomaly"].sum() == sample_results["anomaly"].sum()
    assert np.array_equal(data["known_label_class"], sample_results["Class"])
    unlabelled = pd.read_csv(io.BytesIO(compact_results_csv(sample_results.drop(columns=["Class"]), "x", 1.0)))
    assert "known_label_class" not in unlabelled.columns


@needs_original
def test_case_files_use_only_real_values(sample_results, models):
    threshold = models["baseline"][2]
    positions = flagged_positions(sample_results)
    errors = sample_results["reconstruction_error"].to_numpy()
    assert len(positions) == int(sample_results["anomaly"].sum())
    assert np.all(np.diff(errors[positions]) <= 0)  # highest error first

    position = int(positions[0])
    case = build_case_file(sample_results, position, threshold, "Baseline Autoencoder")
    row = sample_results.iloc[position]
    assert case["reference"] == case_reference(position) == f"IF-{position:06d}"
    assert case["amount"] == row["Amount"]
    assert case["reconstruction_error"] == row["reconstruction_error"]
    assert case["threshold"] == threshold
    assert case["classification"] == "Anomaly" and case["status"] == "ANOMALY FLAGGED"
    assert case["note"] == INVESTIGATION_NOTE
    assert case["known_label"] in ("Fraud (Class = 1)", "Legitimate (Class = 0)")
    assert abs(case["error_to_threshold_ratio"] - row["reconstruction_error"] / threshold) < 1e-12
    assert "confidence" not in " ".join(map(str, case.values())).lower()

    unlabelled = build_case_file(sample_results.drop(columns=["Class"]), position, threshold, "x")
    assert unlabelled["known_label"] == "Not available"


def test_describe_time():
    assert describe_time(0) == "0 s after the first transaction (day 1, 00:00:00)"
    assert describe_time(90_061) == "90,061 s after the first transaction (day 2, 01:01:01)"
    assert describe_time(None) == "Not available"


# ------------------------------------------------------------------
# Official results: read, never recalculated
# ------------------------------------------------------------------

def test_official_comparison_matches_committed_files():
    table, conclusion = load_official_comparison(METRICS / "baseline_metrics.csv", METRICS / "improved_metrics.csv",
                                                 METRICS / "model_comparison_baseline_vs_improved.csv")
    by_key = table.set_index("key")
    assert int(by_key.loc["true_positives", "baseline"]) == 85
    assert int(by_key.loc["false_positives", "baseline"]) == 2_978
    assert int(by_key.loc["true_positives", "improved"]) == 84
    assert conclusion["supported"] is False
    assert conclusion["ap_low"] < 0 < conclusion["ap_high"]
    assert conclusion["test_rows"] == 56_962 and conclusion["test_fraud"] == 98
    for key in ["precision", "recall", "f1_score", "average_precision", "roc_auc", "false_positive_rate"]:
        assert key in by_key.index


# ------------------------------------------------------------------
# Live chart data: computed from the loaded file or stored results only
# ------------------------------------------------------------------

@needs_synthetic
def test_class_distribution_and_amount_statistics(synthetic_df):
    start = du.synthetic_start_row(du.read_synthetic_metadata())
    dist = class_distribution(synthetic_df, start).set_index("group")
    assert dist.loc["Legitimate", "transactions"] == 284_315
    assert dist.loc["Fraud (original)", "transactions"] == 492
    assert dist.loc["Fraud (synthetic)", "transactions"] == 14_472
    assert abs(dist["share"].sum() - 1.0) < 1e-12
    stats = amount_statistics(synthetic_df, start).set_index("Group")
    legit = synthetic_df.loc[synthetic_df["Class"] == 0, "Amount"]
    assert stats.loc["Legitimate", "Median amount"] == legit.median()
    assert stats.loc["Legitimate", "Largest amount"] == legit.max()


def test_class_helpers_without_labels():
    frame = pd.DataFrame({"Amount": [1.0, 3.0]})
    assert class_distribution(frame).empty
    stats = amount_statistics(frame)
    assert list(stats["Group"]) == ["All transactions"] and stats["Median amount"].iloc[0] == 2.0


def test_anomaly_score_bands_match_flagging():
    threshold = 2.0
    errors = np.array([0.1, 1.5, 2.0, 2.0001, 3.0, 9.9, 15.0, 25.0])  # 2.0 = exactly the threshold
    bands = anomaly_score_bands(errors, threshold)
    assert list(bands["band"]) == SCORE_BAND_LABELS
    assert bands["transactions"].sum() == len(errors)
    flagged = bands.loc[bands["result"] == "Anomaly (flagged)", "transactions"].sum()
    assert flagged == int((errors > threshold).sum()) == 5
    assert bands.set_index("band").loc["0.5x - 1x", "transactions"] == 2  # 1.5 and the exact threshold


@needs_original
def test_score_bands_on_real_results(sample_results, models):
    bands = anomaly_score_bands(sample_results["reconstruction_error"], models["baseline"][2])
    flagged = bands.loc[bands["result"] == "Anomaly (flagged)", "transactions"].sum()
    assert flagged == int(sample_results["anomaly"].sum())


@needs_original
def test_case_context(sample_results):
    positions = flagged_positions(sample_results)
    top = case_context(sample_results, int(positions[0]))
    assert top["flagged_rank"] == 1 and top["flagged_total"] == len(positions)
    assert top["share_below"] == (len(sample_results) - 1) / len(sample_results)


@pytest.mark.parametrize("model", ["baseline", "improved"])
def test_pr_curve_from_stored_scores_matches_official_ap(model):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    scores = pd.read_csv(METRICS / f"{model}_test_scores.csv")
    official = pd.read_csv(METRICS / f"{model}_metrics.csv")
    official = official[official["evaluation_set"] == "official_full_test"].set_index("metric")["value"]
    # The stored scores reproduce the official figures exactly, so the drawn curve is the official one.
    assert average_precision_score(scores["actual_class"], scores["reconstruction_error"]) == pytest.approx(
        official["average_precision"], abs=1e-12)
    points = pr_curve_points(scores["actual_class"], scores["reconstruction_error"])
    assert len(points) <= 2_000
    assert points["recall"].max() == 1.0 and points["recall"].min() == 0.0
    _, full_recall, _ = precision_recall_curve(scores["actual_class"], scores["reconstruction_error"])
    assert set(points["recall"]) == set(full_recall)  # every recall step of the full curve is kept


def test_test_set_error_histogram_shares():
    scores = pd.read_csv(METRICS / "baseline_test_scores.csv")
    edges = shared_log_edges([scores["reconstruction_error"]])
    hist = class_error_histogram(scores["reconstruction_error"], scores["actual_class"], edges)
    totals = hist.groupby("group")["count"].sum()
    assert totals["Fraud"] == 98 and totals["Legitimate"] == 56_864
    assert np.allclose(hist.groupby("group")["share"].sum(), 1.0)


def test_confusion_ci_and_paired_frames_come_from_official_files():
    table, conclusion = load_official_comparison(METRICS / "baseline_metrics.csv", METRICS / "improved_metrics.csv",
                                                 METRICS / "model_comparison_baseline_vs_improved.csv")
    cells = confusion_frame(table).set_index(["model", "outcome"])["count"]
    assert [cells["Baseline", k] for k in ["True negative", "False positive", "False negative", "True positive"]] \
        == [53_886, 2_978, 13, 85]
    assert [cells["Improved v2", k] for k in ["True negative", "False positive", "False negative", "True positive"]] \
        == [53_877, 2_987, 14, 84]
    ci = ci_frame(table).set_index(["model", "key"])
    assert ci.loc[("Baseline", "average_precision"), "value"] == conclusion["ap_baseline"]
    paired = paired_difference_frame(table).set_index("key")
    assert paired.loc["average_precision", "primary"]
    assert paired.loc["average_precision", "low"] == conclusion["ap_low"] < 0 < conclusion["ap_high"]


def test_training_histories():
    baseline = training_history(ROOT / "models" / "training_history.csv", "Baseline")
    improved = training_history(ROOT / "models" / "improved" / "training_history.csv", "Improved v2")
    assert baseline["epoch"].max() == 10 and improved["epoch"].max() == 100
    assert set(baseline["split"]) == {"Training loss", "Validation loss"}
    raw = pd.read_csv(ROOT / "models" / "improved" / "training_history.csv")
    assert improved.loc[improved["split"] == "Validation loss", "loss_value"].tolist() == raw["val_loss"].tolist()


# ------------------------------------------------------------------
# The dashboard itself (headless)
# ------------------------------------------------------------------

@needs_original
def test_default_view_identity_and_defaults():
    at = run_app()
    assert not at.exception
    text = page_text(at)
    assert "THE FRAUD BUREAU" in text and "IntelliFraud" in text
    assert "Credit Card Fraud Detection Using Deep Learning-Based Autoencoders" in text
    assert "SYSTEM READY" in text
    assert at.radio(key="dataset_choice").value == "original"
    assert at.radio(key="model_choice").value == "baseline"
    labels = [tab.label for tab in at.tabs]
    for name in ["Dataset Intelligence", "Transaction Investigations", "Bureau Intelligence Report",
                 "Transaction Case Files"]:
        assert name in labels
    assert "Bureau Methodology" not in labels  # removed from the web application
    assert metric_value(at, "Total Transactions") == "284,807"
    assert metric_value(at, "Fraud Percentage") == "0.173%"
    assert len(at.file_uploader) == 1


def chart_count(at):
    return len(at.get("vega_lite_chart"))


@needs_original
def test_live_charts_render_without_static_images():
    at = run_app()
    assert not at.exception
    before = chart_count(at)
    assert before >= 11  # dataset charts + report charts
    assert len(at.get("imgs")) == 0 and len(at.get("image")) == 0  # no saved graph images are shown any more
    at.button(key="detect_button").click().run()
    assert chart_count(at) == before + 3  # error distribution, flagged share, anomaly score bands


@needs_original
def test_report_tab_shows_conclusion_and_comparison():
    text = page_text(run_app())
    assert CONCLUSION in text
    assert "56,962 held-out test transactions" in text
    assert "Why false positives matter" in text
    assert "Isolation Forest" in text  # stored three-model comparison still available


@needs_synthetic
def test_synthetic_dataset_selection_shows_warning_and_stats():
    at = run_app()
    at.radio(key="dataset_choice").set_value("synthetic").run()
    assert not at.exception
    assert metric_value(at, "Total Transactions") == "299,279"
    assert metric_value(at, "Fraud Percentage") == "5.000%"
    warnings = " ".join(str(w.value) for w in at.warning)
    assert "artificially generated fraud examples" in warnings
    assert "not** an independent real-world test dataset" in warnings


@needs_original
def test_improved_model_detection_and_threshold():
    at = run_app()
    at.radio(key="model_choice").set_value("improved").run()
    at.button(key="detect_button").click().run()
    assert not at.exception
    assert metric_value(at, "Transactions Scored") == "284,807"
    assert metric_value(at, "Anomalies Detected") == "15,184"
    assert metric_value(at, "Model Threshold") == "0.9820"
    assert "Improved v2 was trained on" in page_text(at)


@needs_original
def test_switching_model_clears_results():
    at = run_app()
    at.button(key="detect_button").click().run()
    assert metric_value(at, "Anomalies Detected") == "14,769"
    at.radio(key="model_choice").set_value("improved").run()
    assert metric_value(at, "Transactions Scored") is None
    assert "previous results were cleared" in page_text(at)


@needs_original
def test_case_files_downloads_and_label_counts_after_detection():
    at = run_app()
    at.button(key="detect_button").click().run()
    assert not at.exception
    text = page_text(at)
    assert "CASE #IF-" in text and "ANOMALY FLAGGED" in text
    assert "Further review is recommended." in text
    assert "1.073887" in text  # the baseline threshold inside the case file
    assert at.selectbox(key="case_file_choice").value is not None
    download_labels = [element.proto.label for element in at.get("download_button")]
    assert "Download flagged transactions (CSV)" in download_labels
    assert "Download compact full results (CSV)" in download_labels
    assert LABEL_WARNING in " ".join(str(w.value) for w in at.warning)


@needs_original
def test_upload_overrides_selection_and_invalid_upload_errors(original_df):
    at = run_app()
    at.file_uploader[0].upload("mine.csv", original_df.head(500).to_csv(index=False).encode(), "text/csv").run()
    assert "Uploaded dataset in use" in page_text(at)
    assert metric_value(at, "Total Transactions") == "500"

    header_only = ",".join(original_df.columns).encode() + b"\n"
    at.file_uploader[0].upload("empty.csv", header_only, "text/csv").run()
    at.button(key="detect_button").click().run()
    assert not at.exception
    assert "0 rows" in " ".join(str(e.value) for e in at.error)

    at.file_uploader[0].upload("labels_only.csv", b"Class\n0\n1\n", "text/csv").run()
    at.button(key="detect_button").click().run()
    errors = " ".join(str(e.value) for e in at.error)
    assert "cannot be scored" in errors and "missing 30 required column(s)" in errors


def test_missing_dataset_shows_helpful_message(monkeypatch, fresh_cache, tmp_path):
    monkeypatch.setitem(DATASET_OPTIONS["synthetic"], "path", tmp_path / "creditcard_synthetic_5pct.csv")
    at = run_app()
    at.radio(key="dataset_choice").set_value("synthetic").run()
    assert not at.exception
    errors = " ".join(str(e.value) for e in at.error)
    assert "Dataset file not found" in errors and "python src/generate_synthetic_data.py" in errors
    assert at.button(key="detect_button").disabled


@needs_original
def test_missing_improved_model_shows_helpful_message(monkeypatch, fresh_cache, tmp_path):
    monkeypatch.setattr(du, "IMPROVED_DIR", tmp_path / "improved")
    at = run_app()
    at.radio(key="model_choice").set_value("improved").run()
    assert not at.exception
    errors = " ".join(str(e.value) for e in at.error)
    assert "could not be loaded" in errors and "Improved Autoencoder (v2)" in errors
    assert at.button(key="detect_button").disabled
    assert "MODEL OFFLINE" in page_text(at)


# ------------------------------------------------------------------
# Repository hygiene
# ------------------------------------------------------------------

OFFICIAL_FILES = [
    "models/autoencoder.keras", "models/scaler.pkl", "models/threshold.txt", "models/test_indices.csv",
    "models/improved/autoencoder.keras", "models/improved/scaler.pkl", "models/improved/threshold.txt",
    "models/improved/config.json", "models/model_comparison.csv", "models/evaluation_results.csv",
    "results/metrics/baseline_metrics.csv", "results/metrics/improved_metrics.csv",
    "results/metrics/model_comparison_baseline_vs_improved.csv", "results/metrics/baseline_evaluation_report.txt",
    "results/metrics/improved_evaluation_report.txt", "results/graphs/pr_curve_baseline_vs_improved.png",
    "results/graphs/baseline_confusion_matrix.png", "results/graphs/improved_confusion_matrix.png",
    "results/graphs/improved_training_history.png", "results/metrics/baseline_test_scores.csv",
    "results/metrics/improved_test_scores.csv", "models/training_history.csv",
    "models/improved/training_history.csv",
]


@pytest.mark.parametrize("path", OFFICIAL_FILES)
def test_official_models_results_and_graphs_unchanged(path):
    assert (ROOT / path).read_bytes() == git_head_bytes(path)


def test_no_datasets_virtualenv_or_secrets_tracked():
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT).decode().split("\n")
    assert not [f for f in tracked if f.startswith("data/") and f.endswith(".csv")]
    assert not [f for f in tracked if f.startswith((".venv/", "venv/"))]
    assert not [f for f in tracked if f.endswith(("secrets.toml", ".env", ".pem", ".key"))]


# Emoji code-point ranges, written as numbers so this file stays plain ASCII.
EMOJI_RANGES = [(0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x1F000, 0x1F2FF), (0xFE0F, 0xFE0F),
                (0x2B50, 0x2B50), (0x2705, 0x2705), (0x274C, 0x274C)]
EMOJI = re.compile("[" + "".join(f"{re.escape(chr(a))}-{re.escape(chr(b))}" for a, b in EMOJI_RANGES) + "]")


@pytest.mark.parametrize("path", ["app.py", "src/dashboard_utils.py", ".streamlit/config.toml",
                                  "tests/test_dashboard.py", "tests/test_phase1.py"])
def test_dashboard_files_are_emoji_free(path):
    assert not EMOJI.findall((ROOT / path).read_text(encoding="utf-8"))
