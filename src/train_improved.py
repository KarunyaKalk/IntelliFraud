"""
Train the IntelliFraud "Improved v2" Autoencoder.

Same architecture, same 30 input features (same order), same reconstruction
error and the SAME official test set as the original baseline - but a
cleaner training method:

  1. Start from the original training split (all rows NOT in
     models/test_indices.csv). The official test set is never touched.
  2. Keep legitimate transactions only (Class == 0).
  3. Remove repeated copies (identical 30 feature values), keeping one.
  4. Remove any transaction whose 30 feature values exactly match a
     transaction in the official test set (hash match, then full-row check).
  5. Split the cleaned pool 85% training / 15% validation (random_state=42).
  6. Fit StandardScaler on the 85% training rows only.
  7. Train the Autoencoder on the training rows; the validation rows are used
     for early stopping (up to 100 epochs, patience 5, best weights restored).
  8. Threshold = 95th percentile of VALIDATION reconstruction errors.

Fraud labels are used only to exclude fraud rows. The test set is used for
nothing here.

Safety: all files are written to models/improved.partial/ and that folder is
renamed to models/improved/ only after everything succeeded. If either folder
already exists the script stops without writing anything.

Run from the project folder:
    python src/train_improved.py
"""

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Settings (approved experiment design)
# ------------------------------------------------------------------
SEED = 42
VALIDATION_SIZE = 0.15
MAX_EPOCHS = 100
PATIENCE = 5
BATCH_SIZE = 512
THRESHOLD_PERCENTILE = 95

DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
TEST_INDEX_PATH = PROJECT_ROOT / "models" / "test_indices.csv"
BASELINE_SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"  # read only: feature names/order

OUTPUT_DIR = PROJECT_ROOT / "models" / "improved"
STAGING_DIR = PROJECT_ROOT / "models" / "improved.partial"
OUTPUT_FILES = [
    "autoencoder.keras",
    "scaler.pkl",
    "threshold.txt",
    "training_history.csv",
    "config.json",
    "train_indices.csv",
    "validation_indices.csv",
]


class OverwriteError(RuntimeError):
    """Raised instead of overwriting existing files."""


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_no_existing_outputs(output_dir=OUTPUT_DIR, staging_dir=STAGING_DIR):
    """Stop if a previous run left any output behind. Never overwrite."""
    problems = []
    if output_dir.exists():
        existing = sorted(p.name for p in output_dir.iterdir())
        problems.append(f"{output_dir} already exists (contains: {', '.join(existing) or 'nothing'})")
    if staging_dir.exists():
        problems.append(f"{staging_dir} exists (left over from an interrupted run)")
    if problems:
        raise OverwriteError(
            "Refusing to train because existing files would be overwritten: "
            + "; ".join(problems)
            + ". Move or delete them yourself first if you really want a new run."
        )


# ------------------------------------------------------------------
# Data preparation (pure functions - also used by tests/test_phase2.py)
# ------------------------------------------------------------------

def load_inputs():
    """Dataset, official test index (original order) and the baseline feature order."""
    df = pd.read_csv(DATA_PATH)
    test_index = pd.Index(pd.read_csv(TEST_INDEX_PATH)["index"].astype(int))
    feature_names = list(joblib.load(BASELINE_SCALER_PATH).feature_names_in_)
    return df, test_index, feature_names


def rows_matching_reference(features, candidate_index, reference_index):
    """
    Boolean flags: does each candidate row have an EXACT twin (all feature
    values equal) among the reference rows? Hash first, then full comparison,
    so a hash collision can never create a false match.
    """
    row_hash = pd.util.hash_pandas_object(features, index=False)
    values = features.to_numpy()
    position = pd.Series(np.arange(len(features)), index=features.index)

    reference_by_hash = {}
    for idx, h in row_hash.loc[reference_index].items():
        reference_by_hash.setdefault(h, []).append(position[idx])

    flags = np.zeros(len(candidate_index), dtype=bool)
    for i, (idx, h) in enumerate(row_hash.loc[candidate_index].items()):
        matches = reference_by_hash.get(h)
        if matches:
            row = values[position[idx]]
            flags[i] = any(np.array_equal(row, values[j]) for j in matches)
    return flags


def build_clean_pool(df, test_index, feature_names):
    """
    Legitimate training pool after removing duplicates and test-set twins.

    Returns (clean_index, counts). Duplicates are judged on the 30 feature
    values only (Class excluded).
    """
    features = df[feature_names]
    counts = {"dataset_rows": int(len(df)), "official_test_rows": int(len(test_index))}

    in_test = df.index.isin(test_index)
    training_split = df.index[~in_test]
    counts["training_split_rows"] = int(len(training_split))
    counts["training_split_fraud_rows_excluded"] = int((df.loc[training_split, "Class"] == 1).sum())

    legit = training_split[df.loc[training_split, "Class"].to_numpy() == 0]
    counts["legitimate_training_pool_rows"] = int(len(legit))

    repeated = features.loc[legit].duplicated(keep="first").to_numpy()
    counts["repeated_copies_removed"] = int(repeated.sum())
    unique_legit = legit[~repeated]
    counts["after_removing_repeats"] = int(len(unique_legit))

    matches_test = rows_matching_reference(features, unique_legit, test_index)
    counts["test_set_twins_removed"] = int(matches_test.sum())
    clean = unique_legit[~matches_test]
    counts["clean_pool_rows"] = int(len(clean))

    return clean, counts


def split_train_validation(clean_index):
    """85% training / 15% validation of the cleaned legitimate pool."""
    train_index, validation_index = train_test_split(
        clean_index, test_size=VALIDATION_SIZE, random_state=SEED, shuffle=True
    )
    return pd.Index(train_index), pd.Index(validation_index)


def verify_separation(df, feature_names, train_index, validation_index, test_index):
    """Raise if any leakage rule is broken. Returns a dict of checks."""
    features = df[feature_names]
    checks = {
        "no_fraud_in_training": int(df.loc[train_index, "Class"].sum()) == 0,
        "no_fraud_in_validation": int(df.loc[validation_index, "Class"].sum()) == 0,
        "training_validation_disjoint_rows": len(train_index.intersection(validation_index)) == 0,
        "training_validation_no_identical_transactions":
            not rows_matching_reference(features, validation_index, train_index).any(),
        "training_no_test_rows": len(train_index.intersection(test_index)) == 0,
        "validation_no_test_rows": len(validation_index.intersection(test_index)) == 0,
        "training_no_test_twins": not rows_matching_reference(features, train_index, test_index).any(),
        "validation_no_test_twins": not rows_matching_reference(features, validation_index, test_index).any(),
        "no_repeats_in_training_plus_validation":
            not features.loc[train_index.append(validation_index)].duplicated().any(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Leakage checks failed: {failed}")
    return checks


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def main():
    check_no_existing_outputs()

    import tensorflow as tf
    import keras
    from tensorflow.keras.callbacks import EarlyStopping

    from src.anomaly_detection import calculate_reconstruction_error
    from src.autoencoder import build_autoencoder

    tf.keras.utils.set_random_seed(SEED)          # Python, NumPy and TensorFlow seeds
    tf.config.experimental.enable_op_determinism()  # same results on every run (same machine)

    print("\n===== 1. LOADING INPUTS =====")
    df, test_index, feature_names = load_inputs()
    print(f"Dataset rows: {len(df):,} | official test rows: {len(test_index):,} | features: {len(feature_names)}")

    print("\n===== 2. CLEANING THE LEGITIMATE TRAINING POOL =====")
    clean_index, counts = build_clean_pool(df, test_index, feature_names)
    for key, value in counts.items():
        print(f"  {key:<38} {value:>8,}")

    print("\n===== 3. TRAINING / VALIDATION SPLIT (85/15, random_state=42) =====")
    train_index, validation_index = split_train_validation(clean_index)
    counts["training_rows"] = int(len(train_index))
    counts["validation_rows"] = int(len(validation_index))
    print(f"  training rows   {len(train_index):>8,}")
    print(f"  validation rows {len(validation_index):>8,}")
    checks = verify_separation(df, feature_names, train_index, validation_index, test_index)
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("\n===== 4. SCALING (fitted on training rows only) =====")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df.loc[train_index, feature_names])
    X_validation = scaler.transform(df.loc[validation_index, feature_names])
    print(f"  scaler fitted on {int(scaler.n_samples_seen_):,} rows")

    print("\n===== 5. TRAINING THE AUTOENCODER =====")
    model = build_autoencoder(len(feature_names))
    early_stopping = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    start = time.perf_counter()
    history = model.fit(
        X_train, X_train,
        validation_data=(X_validation, X_validation),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=2,
    )
    training_seconds = time.perf_counter() - start
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    epochs_completed = len(history_df)
    best_epoch = int(history_df["val_loss"].idxmin()) + 1
    stopped_early = epochs_completed < MAX_EPOCHS
    print(f"  epochs completed: {epochs_completed} | best epoch: {best_epoch} | "
          f"stopped early: {stopped_early} | time: {training_seconds:.1f} s")

    print("\n===== 6. THRESHOLD (95th percentile of VALIDATION errors) =====")
    validation_errors = calculate_reconstruction_error(model, X_validation)
    threshold = float(np.percentile(validation_errors, THRESHOLD_PERCENTILE))
    training_errors = calculate_reconstruction_error(model, X_train)
    print(f"  threshold: {threshold:.10f}")
    print(f"  validation rows above threshold: {(validation_errors > threshold).mean():.2%}")
    print(f"  training rows above threshold:   {(training_errors > threshold).mean():.2%}")
    print(f"  (training-error 95th percentile would be {np.percentile(training_errors, 95):.4f})")

    print("\n===== 7. SAVING (staging folder, then rename) =====")
    check_no_existing_outputs()  # re-check right before writing
    STAGING_DIR.mkdir(parents=False, exist_ok=False)

    model.save(STAGING_DIR / "autoencoder.keras")
    joblib.dump(scaler, STAGING_DIR / "scaler.pkl")
    (STAGING_DIR / "threshold.txt").write_text(repr(threshold))
    history_df.to_csv(STAGING_DIR / "training_history.csv", index=False)
    pd.Series(train_index, name="index").to_csv(STAGING_DIR / "train_indices.csv", index=False)
    pd.Series(validation_index, name="index").to_csv(STAGING_DIR / "validation_indices.csv", index=False)

    config = {
        "model_name": "Improved v2",
        "description": "Same architecture/features as the baseline; de-duplicated legitimate training "
                       "pool, test-set twins removed, 85/15 train/validation split, early stopping on "
                       "validation, threshold = 95th percentile of validation errors.",
        "random_seed": SEED,
        "op_determinism": True,
        "feature_names": feature_names,
        "feature_order_source": "models/scaler.pkl (baseline) feature_names_in_",
        "architecture": [len(feature_names), 16, 8, 16, len(feature_names)],
        "architecture_source": "src/autoencoder.py build_autoencoder (unchanged)",
        "reconstruction_error": "mean squared error over the 30 scaled features "
                                "(src/anomaly_detection.calculate_reconstruction_error)",
        "training_parameters": {
            "optimizer": "adam", "loss": "mse", "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS, "early_stopping_monitor": "val_loss",
            "patience": PATIENCE, "restore_best_weights": True, "shuffle": True,
        },
        "split": {"validation_size": VALIDATION_SIZE, "random_state": SEED,
                  "duplicate_definition": "identical values in all 30 feature columns (Class excluded)"},
        "row_counts": counts,
        "leakage_checks": checks,
        "threshold": threshold,
        "threshold_rule": f"{THRESHOLD_PERCENTILE}th percentile of validation reconstruction errors",
        "results": {
            "epochs_completed": epochs_completed,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "best_val_loss": float(history_df["val_loss"].min()),
            "training_seconds": round(training_seconds, 1),
            "validation_share_above_threshold": float((validation_errors > threshold).mean()),
            "training_share_above_threshold": float((training_errors > threshold).mean()),
        },
        "input_fingerprints": {
            "data/creditcard.csv": sha256_of(DATA_PATH),
            "models/test_indices.csv": sha256_of(TEST_INDEX_PATH),
            "models/scaler.pkl": sha256_of(BASELINE_SCALER_PATH),
        },
        "software": {
            "python": platform.python_version(), "tensorflow": tf.__version__,
            "keras": keras.__version__, "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "machine": platform.machine(),
        },
    }
    (STAGING_DIR / "config.json").write_text(json.dumps(config, indent=2))

    written = sorted(p.name for p in STAGING_DIR.iterdir())
    if written != sorted(OUTPUT_FILES):
        raise RuntimeError(f"Unexpected staging contents {written}; left in {STAGING_DIR} for inspection.")
    STAGING_DIR.rename(OUTPUT_DIR)  # atomic: models/improved appears complete or not at all

    print(f"\nSaved to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/: {', '.join(OUTPUT_FILES)}")
    print("===== TRAINING COMPLETED =====")


if __name__ == "__main__":
    main()
