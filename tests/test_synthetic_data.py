"""
Tests for the synthetic 5%-fraud dataset (src/generate_synthetic_data.py).

Run from the project folder:
    python -m pytest tests/test_synthetic_data.py -v

Tests that need the generated file are skipped until it exists. The
reproducibility and overwrite tests generate data in memory / in a temporary
folder only and never touch the real outputs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.generate_synthetic_data import (
    COLUMNS,
    FEATURES,
    OverwriteError,
    check_outputs_absent,
    generate_synthetic_fraud,
    load_original,
    sha256_of,
    synthetic_rows_as_csv_text,
    target_counts,
    validate_synthetic,
    write_synthetic_csv,
)

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data" / "creditcard.csv"
SYNTHETIC = ROOT / "data" / "synthetic" / "creditcard_synthetic_5pct.csv"
METADATA = ROOT / "data" / "synthetic" / "synthetic_data_metadata.json"

# Fingerprint of the untouched Kaggle file (also recorded in results/metrics/dataset_report.txt)
ORIGINAL_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"

needs_original = pytest.mark.skipif(not ORIGINAL.exists(), reason="data/creditcard.csv not present")
needs_synthetic = pytest.mark.skipif(not SYNTHETIC.exists(), reason="synthetic dataset not generated yet")


@pytest.fixture(scope="module")
def original():
    return load_original()


@pytest.fixture(scope="module")
def synthetic_file():
    return pd.read_csv(SYNTHETIC, float_precision="round_trip")


@pytest.fixture(scope="module")
def counts(original):
    return target_counts(original)


# ------------------------------------------------------------------
# 1-6: shape, rate, columns, values
# ------------------------------------------------------------------

@needs_original
def test_target_counts_keep_every_original_row(counts):
    assert counts["original_rows"] == 284_807
    assert counts["original_legitimate"] == 284_315 and counts["original_fraud"] == 492
    assert counts["final_legitimate"] == 284_315          # nothing removed
    assert counts["final_fraud"] == 14_964 and counts["synthetic_fraud"] == 14_472
    assert counts["final_rows"] == 299_279


@needs_original
@needs_synthetic
def test_1_dimensions(synthetic_file, counts):
    assert synthetic_file.shape == (counts["final_rows"], 31)


@needs_original
@needs_synthetic
def test_2_fraud_percentage(synthetic_file, counts):
    rate = synthetic_file["Class"].mean()
    assert abs(rate - 0.05) < 1e-4
    assert int(synthetic_file["Class"].sum()) == counts["final_fraud"]
    assert set(synthetic_file["Class"].unique()) == {0, 1}


@needs_synthetic
def test_3_feature_names_and_order(synthetic_file):
    assert list(synthetic_file.columns) == COLUMNS
    header = SYNTHETIC.open().readline()
    assert header == ORIGINAL.open().readline()


@needs_synthetic
def test_4_no_missing_or_infinite_values(synthetic_file):
    assert not synthetic_file.isna().any().any()
    assert not np.isinf(synthetic_file[FEATURES].to_numpy(dtype=float)).any()


@needs_synthetic
def test_5_no_negative_amount(synthetic_file):
    assert (synthetic_file["Amount"] >= 0).all()


@needs_original
@needs_synthetic
def test_6_time_within_original_range(synthetic_file, original):
    assert synthetic_file["Time"].min() >= original["Time"].min()
    assert synthetic_file["Time"].max() <= original["Time"].max()


# ------------------------------------------------------------------
# 7-9: original rows preserved, generated rows labelled fraud and new
# ------------------------------------------------------------------

@needs_original
@needs_synthetic
def test_7_all_original_legitimate_rows_preserved(synthetic_file, original):
    head = synthetic_file.iloc[: len(original)]
    legit = original["Class"] == 0
    assert head[legit.to_numpy()].reset_index(drop=True).equals(original[legit].reset_index(drop=True))
    assert int((synthetic_file["Class"] == 0).sum()) == int(legit.sum())


@needs_original
@needs_synthetic
def test_8_all_original_fraud_rows_preserved(synthetic_file, original):
    head = synthetic_file.iloc[: len(original)]
    fraud = original["Class"] == 1
    assert head[fraud.to_numpy()].reset_index(drop=True).equals(original[fraud].reset_index(drop=True))


@needs_original
@needs_synthetic
def test_7_8_file_begins_with_the_original_file_byte_for_byte():
    original_bytes = ORIGINAL.read_bytes()
    with open(SYNTHETIC, "rb") as file:
        assert file.read(len(original_bytes)) == original_bytes


@needs_original
@needs_synthetic
def test_9_generated_rows_are_fraud_and_distinct(synthetic_file, original, counts):
    tail = synthetic_file.iloc[len(original):]
    assert len(tail) == counts["synthetic_fraud"]
    assert (tail["Class"] == 1).all()
    # no generated row equals any original row on the 30 features (independent merge check)
    overlap = tail[FEATURES].merge(original[FEATURES].drop_duplicates(), on=FEATURES, how="inner")
    assert len(overlap) == 0
    assert not tail[FEATURES].duplicated().any()


# ------------------------------------------------------------------
# 10: original dataset unchanged
# ------------------------------------------------------------------

@needs_original
def test_10_original_dataset_unchanged():
    assert sha256_of(ORIGINAL) == ORIGINAL_SHA256


@needs_synthetic
def test_10_metadata_records_original_fingerprint_and_passing_validation():
    metadata = json.loads(METADATA.read_text())
    assert metadata["original_dataset"]["sha256"] == ORIGINAL_SHA256
    assert metadata["validation"]["all_passed"] is True
    assert metadata["random_seed"] == 42
    assert "ARTIFICIAL" in metadata["artificial_data_statement"]
    assert metadata["output_file_sha256"] == sha256_of(SYNTHETIC)


# ------------------------------------------------------------------
# 11: reproducibility with seed 42
# ------------------------------------------------------------------

@needs_original
def test_11_same_seed_same_rows_different_seed_different_rows(original, counts):
    a, source_a, _ = generate_synthetic_fraud(original, counts["synthetic_fraud"], seed=42)
    b, source_b, _ = generate_synthetic_fraud(original, counts["synthetic_fraud"], seed=42)
    c, _, _ = generate_synthetic_fraud(original, counts["synthetic_fraud"], seed=7)
    assert a.equals(b) and np.array_equal(source_a, source_b)
    assert not a.equals(c)


@needs_original
@needs_synthetic
def test_11_file_matches_fresh_generation_byte_for_byte(original, counts):
    fresh, _, _ = generate_synthetic_fraud(original, counts["synthetic_fraud"], seed=42)
    expected = ORIGINAL.read_bytes() + synthetic_rows_as_csv_text(fresh).encode("ascii")
    assert SYNTHETIC.read_bytes() == expected


@needs_original
def test_validation_catches_problems(original, counts):
    synthetic, _, _ = generate_synthetic_fraud(original, counts["synthetic_fraud"])
    good = pd.concat([original, synthetic], ignore_index=True)
    checks, _ = validate_synthetic(good, original, expected=counts)
    assert all(checks.values())

    bad = good.copy()
    bad.loc[len(original) + 5, "Amount"] = -1.0
    bad.loc[3, "V1"] += 1e-9                      # tamper with an original legitimate row
    bad.loc[len(original) + 6, FEATURES] = original.loc[original.Class == 1, FEATURES].iloc[0].to_numpy()
    checks, _ = validate_synthetic(bad, original, expected=counts)
    assert not checks["amount_nonnegative"]
    assert not checks["all_original_legitimate_rows_preserved"]
    assert not checks["generated_rows_distinct_from_all_original_rows"]


# ------------------------------------------------------------------
# 12: existing outputs are never overwritten
# ------------------------------------------------------------------

def test_12_refuses_when_outputs_exist(tmp_path):
    existing = tmp_path / "creditcard_synthetic_5pct.csv"
    existing.write_text("do not overwrite")
    with pytest.raises(OverwriteError):
        check_outputs_absent([existing, tmp_path / "metadata.json"])
    leftover = tmp_path / "other.csv"
    Path(str(leftover) + ".partial").write_text("left over")
    with pytest.raises(OverwriteError):
        check_outputs_absent([leftover])
    assert existing.read_text() == "do not overwrite"


@needs_original
def test_12_writer_never_overwrites_a_temporary_file(tmp_path, original):
    target = tmp_path / "out.csv"
    Path(str(target) + ".partial").write_text("existing")
    synthetic, _, _ = generate_synthetic_fraud(original, 5)
    with pytest.raises(FileExistsError):
        write_synthetic_csv(ORIGINAL, synthetic, target)
    assert Path(str(target) + ".partial").read_text() == "existing"


@needs_synthetic
def test_12_running_the_generator_again_is_refused():
    with pytest.raises(OverwriteError):
        check_outputs_absent()
