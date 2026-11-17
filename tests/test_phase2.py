"""
Phase 2 tests: Improved v2 Autoencoder (training data separation, scaler,
threshold, evaluation rows) and protection of the original baseline.

Run from the project folder:
    python -m pytest tests/ -v

* Data-separation tests run without a trained model.
* Tests needing models/improved/ or the improved_* result files are skipped
  until those exist.
* Leakage is checked with an INDEPENDENT method (pandas merge on all 30
  feature values), not with the hashing code used by the training script.
"""

import io
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.train_improved import (
    OverwriteError as TrainOverwriteError,
    build_clean_pool,
    check_no_existing_outputs,
    load_inputs,
    split_train_validation,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
IMPROVED_DIR = ROOT / "models" / "improved"
METRICS = ROOT / "results" / "metrics"

needs_data = pytest.mark.skipif(not DATA_PATH.exists(), reason="data/creditcard.csv not present")
needs_model = pytest.mark.skipif(not (IMPROVED_DIR / "config.json").exists(),
                                 reason="Improved v2 not trained yet (models/improved/ missing)")
needs_eval = pytest.mark.skipif(not (METRICS / "improved_test_scores.csv").exists(),
                                reason="Improved v2 not evaluated yet")


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def prepared():
    df, test_index, features = load_inputs()
    clean, counts = build_clean_pool(df, test_index, features)
    train_index, validation_index = split_train_validation(clean)
    return {"df": df, "test": test_index, "features": features, "clean": clean, "counts": counts,
            "train": train_index, "validation": validation_index}


def exact_overlap(df, features, index_a, index_b):
    """Number of rows in A whose 30 feature values exactly equal some row in B (independent method)."""
    a = df.loc[index_a, features].drop_duplicates()
    b = df.loc[index_b, features].drop_duplicates()
    return len(a.merge(b, on=features, how="inner"))


def git_head_bytes(path):
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)


# ============================================================
# 1-5: data separation (no trained model needed)
# ============================================================

@needs_data
def test_1_no_fraud_in_training_or_validation(prepared):
    df = prepared["df"]
    assert int(df.loc[prepared["train"], "Class"].sum()) == 0
    assert int(df.loc[prepared["validation"], "Class"].sum()) == 0


@needs_data
def test_2_training_and_validation_do_not_overlap(prepared):
    assert len(prepared["train"].intersection(prepared["validation"])) == 0
    assert exact_overlap(prepared["df"], prepared["features"], prepared["train"], prepared["validation"]) == 0
    assert len(prepared["train"]) + len(prepared["validation"]) == len(prepared["clean"])


@needs_data
def test_3_no_exact_overlap_training_vs_test(prepared):
    assert len(prepared["train"].intersection(prepared["test"])) == 0
    assert exact_overlap(prepared["df"], prepared["features"], prepared["train"], prepared["test"]) == 0


@needs_data
def test_4_no_exact_overlap_validation_vs_test(prepared):
    assert len(prepared["validation"].intersection(prepared["test"])) == 0
    assert exact_overlap(prepared["df"], prepared["features"], prepared["validation"], prepared["test"]) == 0


@needs_data
def test_5_clean_pool_has_no_duplicates_and_counts_add_up(prepared):
    df, features, c = prepared["df"], prepared["features"], prepared["counts"]
    assert not df.loc[prepared["clean"], features].duplicated().any()
    assert c["training_split_rows"] + c["official_test_rows"] == c["dataset_rows"]
    assert c["legitimate_training_pool_rows"] + c["training_split_fraud_rows_excluded"] == c["training_split_rows"]
    assert c["legitimate_training_pool_rows"] - c["repeated_copies_removed"] == c["after_removing_repeats"]
    assert c["after_removing_repeats"] - c["test_set_twins_removed"] == c["clean_pool_rows"]
    # 85 / 15 split
    assert abs(len(prepared["validation"]) / len(prepared["clean"]) - 0.15) < 1e-4


@needs_data
def test_split_is_reproducible(prepared):
    again_train, again_validation = split_train_validation(prepared["clean"])
    assert again_train.equals(prepared["train"]) and again_validation.equals(prepared["validation"])


# ============================================================
# 8: official test indices unchanged
# ============================================================

def test_8_official_test_indices_unchanged():
    assert (ROOT / "models" / "test_indices.csv").read_bytes() == git_head_bytes("models/test_indices.csv")
    test_index = pd.read_csv(ROOT / "models" / "test_indices.csv")["index"].to_numpy()
    baseline = pd.read_csv(ROOT / "results" / "metrics" / "baseline_test_scores.csv")
    assert len(test_index) == 56_962
    assert np.array_equal(test_index, baseline["test_row_index"].to_numpy())


# ============================================================
# Overwrite protection
# ============================================================

def test_training_refuses_to_overwrite(tmp_path):
    existing = tmp_path / "improved"
    existing.mkdir()
    (existing / "autoencoder.keras").write_text("placeholder")
    with pytest.raises(TrainOverwriteError):
        check_no_existing_outputs(output_dir=existing, staging_dir=tmp_path / "staging")
    leftover = tmp_path / "improved.partial"
    leftover.mkdir()
    with pytest.raises(TrainOverwriteError):
        check_no_existing_outputs(output_dir=tmp_path / "absent", staging_dir=leftover)


def test_evaluation_refuses_to_overwrite(tmp_path):
    from src.evaluate_improved import OverwriteError as EvalOverwriteError, check_outputs_absent

    existing = tmp_path / "improved_metrics.csv"
    existing.write_text("x")
    import src.evaluate_improved as ev
    original_root = ev.PROJECT_ROOT
    try:
        ev.PROJECT_ROOT = tmp_path
        with pytest.raises(EvalOverwriteError):
            check_outputs_absent([existing])
    finally:
        ev.PROJECT_ROOT = original_root


# ============================================================
# 6-7: trained model artefacts
# ============================================================

@needs_data
@needs_model
def test_saved_indices_and_counts_match_recomputed(prepared):
    saved_train = pd.read_csv(IMPROVED_DIR / "train_indices.csv")["index"].to_numpy()
    saved_validation = pd.read_csv(IMPROVED_DIR / "validation_indices.csv")["index"].to_numpy()
    assert np.array_equal(saved_train, prepared["train"].to_numpy())
    assert np.array_equal(saved_validation, prepared["validation"].to_numpy())
    config = json.loads((IMPROVED_DIR / "config.json").read_text())
    for key, value in prepared["counts"].items():
        assert config["row_counts"][key] == value
    assert config["row_counts"]["training_rows"] == len(saved_train)
    assert config["row_counts"]["validation_rows"] == len(saved_validation)
    assert config["random_seed"] == 42 and all(config["leakage_checks"].values())


@needs_data
@needs_model
def test_6_scaler_fitted_on_training_rows_only(prepared):
    scaler = joblib.load(IMPROVED_DIR / "scaler.pkl")
    features = prepared["features"]
    assert list(scaler.feature_names_in_) == features  # same order as the baseline
    assert int(scaler.n_samples_seen_) == len(prepared["train"])
    train_values = prepared["df"].loc[prepared["train"], features]
    np.testing.assert_allclose(scaler.mean_, train_values.mean().to_numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(scaler.var_, train_values.var(ddof=0).to_numpy(), rtol=1e-8)
    # and NOT on training + validation
    both = prepared["df"].loc[prepared["train"].append(prepared["validation"]), features]
    assert not np.allclose(scaler.mean_, both.mean().to_numpy(), rtol=1e-10, atol=1e-12)


@needs_data
@needs_model
def test_7_threshold_reproducible_from_validation_errors(prepared):
    from src.anomaly_detection import calculate_reconstruction_error, load_model_components

    model, scaler, threshold = load_model_components(
        IMPROVED_DIR / "autoencoder.keras", IMPROVED_DIR / "scaler.pkl", IMPROVED_DIR / "threshold.txt")
    X_validation = scaler.transform(prepared["df"].loc[prepared["validation"], prepared["features"]])
    recomputed = float(np.percentile(calculate_reconstruction_error(model, X_validation), 95))
    assert abs(recomputed - threshold) / threshold <= 1e-5


@needs_model
def test_same_architecture_as_baseline():
    from src.anomaly_detection import load_model_components
    from src.dashboard_utils import architecture_text

    baseline = load_model_components()[0]
    improved = load_model_components(
        IMPROVED_DIR / "autoencoder.keras", IMPROVED_DIR / "scaler.pkl", IMPROVED_DIR / "threshold.txt")[0]
    assert architecture_text(improved) == architecture_text(baseline) == "30 → 16 → 8 → 16 → 30"


# ============================================================
# 9: evaluation on exactly the baseline's test rows
# ============================================================

@needs_data
@needs_model
@needs_eval
def test_9_evaluation_uses_exactly_the_baseline_test_rows():
    improved = pd.read_csv(METRICS / "improved_test_scores.csv", float_precision="round_trip")
    baseline = pd.read_csv(METRICS / "baseline_test_scores.csv", float_precision="round_trip")
    test_index = pd.read_csv(ROOT / "models" / "test_indices.csv")["index"].to_numpy()
    assert np.array_equal(improved["test_row_index"].to_numpy(), test_index)
    assert np.array_equal(improved["test_row_index"].to_numpy(), baseline["test_row_index"].to_numpy())
    assert np.array_equal(improved["actual_class"].to_numpy(), baseline["actual_class"].to_numpy())

    from src.anomaly_detection import detect_anomalies, load_model_components
    df = pd.read_csv(DATA_PATH)
    components = load_model_components(
        IMPROVED_DIR / "autoencoder.keras", IMPROVED_DIR / "scaler.pkl", IMPROVED_DIR / "threshold.txt")
    rescored = detect_anomalies(df.loc[test_index], components)
    assert np.array_equal(rescored["reconstruction_error"].to_numpy(), improved["reconstruction_error"].to_numpy())
    assert np.array_equal(rescored["anomaly"].to_numpy(), improved["predicted_class"].to_numpy())


@needs_model
@needs_eval
def test_report_states_training_length_and_early_stopping_correctly():
    config = json.loads((IMPROVED_DIR / "config.json").read_text())
    report = (METRICS / "improved_evaluation_report.txt").read_text()
    r = config["results"]
    if not r["stopped_early"]:
        assert f"Training reached the maximum of {config['training_parameters']['max_epochs']} epochs" in report
        assert "early stopping did not trigger" in report
        assert "Validation loss was still decreasing slowly" in report
        assert "no additional training was performed" in report
    assert f"occurred at epoch {r['best_epoch']}" in report
    source = (ROOT / "src" / "evaluate_improved.py").read_text()
    assert "early stopping on validation loss" not in source  # old misleading graph title


@needs_eval
def test_decision_follows_the_preregistered_rule():
    comparison = pd.read_csv(METRICS / "model_comparison_baseline_vs_improved.csv").set_index("metric")
    ap = comparison.loc["average_precision"]
    supported = bool(comparison.loc["improvement_statistically_supported",
                                    "difference_improved_minus_baseline"])
    assert supported == (ap["paired_ci95_low"] > 0)
    assert ap["paired_ci95_low"] <= ap["difference_improved_minus_baseline"] <= ap["paired_ci95_high"]


# ============================================================
# 10: baseline model and result files unchanged
# ============================================================

BASELINE_FILES = [
    "models/autoencoder.keras", "models/scaler.pkl", "models/threshold.txt", "models/test_indices.csv",
    "models/training_history.csv", "models/evaluation_results.csv", "models/model_comparison.csv",
    "results/metrics/baseline_evaluation_report.txt", "results/metrics/baseline_metrics.csv",
    "results/metrics/baseline_test_scores.csv", "results/metrics/dataset_report.txt",
    "results/graphs/baseline_pr_curve.png", "results/graphs/baseline_reconstruction_error_distribution.png",
    "results/graphs/baseline_confusion_matrix.png", "results/graphs/class_distribution.png",
    "results/graphs/amount_distribution.png", "results/graphs/amount_by_class.png",
]


@pytest.mark.parametrize("path", BASELINE_FILES)
def test_10_baseline_files_identical_to_committed_version(path):
    assert (ROOT / path).read_bytes() == git_head_bytes(path), f"{path} differs from the committed version"
