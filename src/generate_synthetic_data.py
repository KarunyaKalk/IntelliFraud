"""
Generate a SYNTHETIC IntelliFraud dataset with a 5% fraud rate.

What it produces
----------------
data/synthetic/creditcard_synthetic_5pct.csv
    ALL 284,807 original transactions, byte-for-byte unchanged (same order),
    followed by newly generated fraudulent transactions, so that fraud makes
    up 5% of the rows. Same 31 columns, same names, same order.
data/synthetic/synthetic_data_metadata.json
    Method, formula, parameters, seed, counts, validation results.
results/graphs/synthetic_class_distribution.png
results/graphs/synthetic_amount_distribution.png

How synthetic fraud rows are made (controlled perturbation)
-----------------------------------------------------------
For each new row:
  1. Pick a SOURCE row at random (with replacement) from the 492 original
     fraud rows (Class == 1). Random generator: numpy default_rng(seed=42).
  2. V1..V28 : x' = x + EPSILON * sigma_j * z      (z ~ Normal(0, 1))
  3. Amount  : a' = round(a * exp(EPSILON * z), 2) (never negative)
  4. Time    : t' = clip(round(t + EPSILON * sigma_time * z), t_min, t_max)
  5. Class   = 1
  sigma_j / sigma_time = standard deviation (ddof=1) of that column over the
  492 ORIGINAL fraud rows; t_min / t_max = observed Time range of the whole
  original dataset. Every z is an independent standard-normal draw.

The new rows are ARTIFICIAL variations of real fraud rows. V1..V28 are
anonymised PCA components, so the changes do not represent any specific
real-world fraud technique.

Safety
------
* data/creditcard.csv is only read; its fingerprint is checked before/after.
* The script stops without writing anything if any output already exists.
* The CSV is written to a temporary file and renamed only after validation.

Run from the project folder:
    python src/generate_synthetic_data.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
SEED = 42
TARGET_FRAUD_RATE = 0.05
EPSILON = 0.05  # perturbation size, as a fraction of the fraud-class standard deviation

ORIGINAL_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
SYNTHETIC_PATH = SYNTHETIC_DIR / "creditcard_synthetic_5pct.csv"
METADATA_PATH = SYNTHETIC_DIR / "synthetic_data_metadata.json"
CLASS_PNG = PROJECT_ROOT / "results" / "graphs" / "synthetic_class_distribution.png"
AMOUNT_PNG = PROJECT_ROOT / "results" / "graphs" / "synthetic_amount_distribution.png"
OUTPUTS = [SYNTHETIC_PATH, METADATA_PATH, CLASS_PNG, AMOUNT_PNG]

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
V_FEATURES = [f"V{i}" for i in range(1, 29)]
COLUMNS = FEATURES + ["Class"]


class OverwriteError(RuntimeError):
    """Raised instead of overwriting an existing output."""


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_outputs_absent(paths=None):
    paths = OUTPUTS if paths is None else paths
    existing = [str(p) for p in paths if Path(p).exists()]
    partial = [str(p) for p in paths if Path(str(p) + ".partial").exists()]
    if existing or partial:
        raise OverwriteError(
            "Refusing to generate because output files already exist: "
            + ", ".join(existing + partial) + ". Move or delete them yourself first."
        )


def load_original(path=ORIGINAL_PATH):
    """Exact float parsing so values can be compared bit-for-bit."""
    return pd.read_csv(path, float_precision="round_trip")


# ------------------------------------------------------------------
# 1. Counts
# ------------------------------------------------------------------

def target_counts(original):
    """
    Keep every original row; add synthetic fraud until fraud = 5% of all rows.
      final_fraud = round(rate * legit / (1 - rate))
    """
    legit = int((original["Class"] == 0).sum())
    fraud = int((original["Class"] == 1).sum())
    final_fraud = int(round(TARGET_FRAUD_RATE * legit / (1 - TARGET_FRAUD_RATE)))
    return {
        "original_rows": int(len(original)),
        "original_legitimate": legit,
        "original_fraud": fraud,
        "synthetic_fraud": final_fraud - fraud,
        "final_fraud": final_fraud,
        "final_legitimate": legit,
        "final_rows": legit + final_fraud,
        "final_fraud_percentage": 100 * final_fraud / (legit + final_fraud),
    }


# ------------------------------------------------------------------
# 2. Generation (pure function: same inputs + seed -> same output)
# ------------------------------------------------------------------

def generate_synthetic_fraud(original, n_synthetic, seed=SEED, epsilon=EPSILON):
    """
    Returns (synthetic DataFrame with COLUMNS, source row indices, parameters).
    Class is never used as an input to the perturbation; it is only used to
    select the source rows (Class == 1) and to label the output (Class = 1).
    """
    rng = np.random.default_rng(seed)
    fraud = original[original["Class"] == 1]

    sigma = fraud[FEATURES].std(ddof=1)
    time_min, time_max = float(original["Time"].min()), float(original["Time"].max())
    time_is_whole_seconds = bool(np.all(original["Time"] == np.round(original["Time"])))

    source_positions = rng.integers(0, len(fraud), size=n_synthetic)
    source = fraud.iloc[source_positions]
    source_index = source.index.to_numpy()

    z_v = rng.standard_normal((n_synthetic, len(V_FEATURES)))
    z_amount = rng.standard_normal(n_synthetic)
    z_time = rng.standard_normal(n_synthetic)

    new = pd.DataFrame(index=np.arange(n_synthetic), columns=COLUMNS, dtype=float)
    new[V_FEATURES] = source[V_FEATURES].to_numpy() + epsilon * sigma[V_FEATURES].to_numpy() * z_v
    new["Amount"] = np.round(source["Amount"].to_numpy() * np.exp(epsilon * z_amount), 2)
    raw_time = source["Time"].to_numpy() + epsilon * sigma["Time"] * z_time
    if time_is_whole_seconds:
        raw_time = np.round(raw_time)
    new["Time"] = np.clip(raw_time, time_min, time_max)
    new["Class"] = 1
    new["Class"] = new["Class"].astype(int)

    parameters = {
        "seed": seed,
        "epsilon": epsilon,
        "random_generator": "numpy.random.default_rng(seed); draws in this order: source rows "
                            "(integers), z for V1..V28 (n x 28), z for Amount (n), z for Time (n)",
        "sigma_source": "standard deviation (ddof=1) over the original fraud rows",
        "sigma": {column: float(sigma[column]) for column in FEATURES},
        "time_range": [time_min, time_max],
        "time_rounded_to_whole_seconds": time_is_whole_seconds,
        "time_values_clipped": int(np.sum((raw_time < time_min) | (raw_time > time_max))),
        "amount_rounded_to_decimals": 2,
    }
    return new, source_index, parameters


# ------------------------------------------------------------------
# 3. Writing: original bytes unchanged + appended synthetic rows
# ------------------------------------------------------------------

def _format_number(value):
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def synthetic_rows_as_csv_text(synthetic):
    """Same style as the original file: unquoted numbers, Class quoted ("1")."""
    lines = []
    for row in synthetic[FEATURES].itertuples(index=False):
        lines.append(",".join(_format_number(v) for v in row) + ',"1"\n')
    return "".join(lines)


def write_synthetic_csv(original_path, synthetic, target_path):
    """Copy the original file byte-for-byte, then append the synthetic rows."""
    original_bytes = Path(original_path).read_bytes()
    if not original_bytes.endswith(b"\n"):
        original_bytes += b"\n"
    temporary = Path(str(target_path) + ".partial")
    with open(temporary, "xb") as file:  # "x": fail instead of overwriting
        file.write(original_bytes)
        file.write(synthetic_rows_as_csv_text(synthetic).encode("ascii"))
    return temporary


# ------------------------------------------------------------------
# 4. Validation (separate function, also used by the tests)
# ------------------------------------------------------------------

def validate_synthetic(combined, original, synthetic_file=None, original_path=ORIGINAL_PATH,
                       original_sha256=None, expected=None):
    """
    Check a combined (original + synthetic) DataFrame. Returns {check: bool}
    plus a separate dict of informational values.
    """
    n_original = len(original)
    expected = expected or target_counts(original)
    feature_values = combined[FEATURES]
    head = combined.iloc[:n_original].reset_index(drop=True)
    tail = combined.iloc[n_original:].reset_index(drop=True)

    checks = {
        "row_count": len(combined) == expected["final_rows"],
        "column_count_31": combined.shape[1] == 31,
        "column_names_and_order": list(combined.columns) == list(original.columns) == COLUMNS,
        "class_only_0_and_1": set(combined["Class"].unique()) <= {0, 1},
        "fraud_rate_approximately_5pct": abs(combined["Class"].mean() - TARGET_FRAUD_RATE) < 1e-4,
        "fraud_count_matches_target": int(combined["Class"].sum()) == expected["final_fraud"],
        "all_original_legitimate_rows_preserved":
            head[original["Class"].to_numpy() == 0].reset_index(drop=True).equals(
                original[original["Class"] == 0].reset_index(drop=True)),
        "all_original_fraud_rows_preserved":
            head[original["Class"].to_numpy() == 1].reset_index(drop=True).equals(
                original[original["Class"] == 1].reset_index(drop=True)),
        "original_rows_in_original_order": head.equals(original.reset_index(drop=True)),
        "generated_rows_all_class_1": bool((tail["Class"] == 1).all()) and len(tail) == expected["synthetic_fraud"],
        "no_missing_values": not combined.isna().any().any(),
        "no_infinite_values": not np.isinf(feature_values.to_numpy(dtype=float)).any(),
        "amount_nonnegative": bool((combined["Amount"] >= 0).all()),
        "time_within_original_range": bool(combined["Time"].between(original["Time"].min(),
                                                                      original["Time"].max()).all()),
    }

    # Distinguishability: no generated row may equal ANY original row on the 30 features
    original_keys = set(pd.util.hash_pandas_object(original[FEATURES], index=False))
    tail_hash = pd.util.hash_pandas_object(tail[FEATURES], index=False)
    hash_matches = int(tail_hash.isin(original_keys).sum())
    exact_matches = 0
    if hash_matches:
        merged = tail[FEATURES].merge(original[FEATURES].drop_duplicates(), on=FEATURES, how="inner")
        exact_matches = len(merged)
    checks["generated_rows_distinct_from_all_original_rows"] = exact_matches == 0
    checks["no_duplicates_among_generated_rows"] = not tail[FEATURES].duplicated().any()

    if synthetic_file is not None:
        original_bytes = Path(original_path).read_bytes()
        with open(synthetic_file, "rb") as file:
            prefix = file.read(len(original_bytes))
        checks["file_starts_with_original_file_bytes"] = prefix == original_bytes
    if original_sha256 is not None:
        checks["original_dataset_unchanged"] = sha256_of(original_path) == original_sha256

    checks = {name: bool(ok) for name, ok in checks.items()}  # plain bools (JSON-safe)

    info = {
        "duplicate_rows_in_whole_dataset_all_31_columns": int(combined.duplicated().sum()),
        "duplicate_rows_in_original_part": int(head.duplicated().sum()),
        "duplicate_rows_in_generated_part": int(tail.duplicated().sum()),
        "generated_rows_hash_matching_original": hash_matches,
        "generated_rows_exactly_equal_to_an_original_row": exact_matches,
        "final_fraud_percentage": float(100 * combined["Class"].mean()),
        "amount_max_original": float(original["Amount"].max()),
        "amount_max_generated": float(tail["Amount"].max()) if len(tail) else None,
    }
    return checks, info


# ------------------------------------------------------------------
# 5. Charts (clearly labelled as synthetic)
# ------------------------------------------------------------------

def make_charts(original, synthetic, counts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    legit_color, orig_fraud_color, synth_fraud_color = "#4C72B0", "#D1495B", "#F2A541"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, log in zip(axes, [False, True]):
        ax.bar(["Legitimate (0)"], [counts["final_legitimate"]], color=legit_color, label="Original legitimate")
        ax.bar(["Fraud (1)"], [counts["original_fraud"]], color=orig_fraud_color, label="Original fraud")
        ax.bar(["Fraud (1)"], [counts["synthetic_fraud"]], bottom=[counts["original_fraud"]],
               color=synth_fraud_color, label="Synthetic fraud (generated)")
        total = counts["final_rows"]
        ax.annotate(f"{counts['final_legitimate']:,}\n({counts['final_legitimate'] / total:.2%})",
                    (0, counts["final_legitimate"]), ha="center", va="bottom", fontsize=9)
        ax.annotate(f"{counts['final_fraud']:,}\n({counts['final_fraud'] / total:.2%})\n"
                    f"= {counts['original_fraud']} original + {counts['synthetic_fraud']:,} synthetic",
                    (1, counts["final_fraud"]), ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Number of transactions" + (" (log scale)" if log else ""))
        if log:
            ax.set_yscale("log")
            ax.set_ylim(top=counts["final_legitimate"] * 12)
        else:
            ax.set_ylim(top=counts["final_legitimate"] * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    axes[0].set_title("Class distribution (linear scale)")
    axes[1].set_title("Class distribution (log scale)")
    fig.suptitle(f"SYNTHETIC DATASET ({counts['final_fraud_percentage']:.2f}% fraud) - not real-world data",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(CLASS_PNG, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, np.log10(max(original["Amount"].max(), synthetic["Amount"].max()) + 1), 50)
    groups = [
        ("Original legitimate", np.log10(original.loc[original.Class == 0, "Amount"] + 1), legit_color),
        ("Original fraud", np.log10(original.loc[original.Class == 1, "Amount"] + 1), orig_fraud_color),
        ("Synthetic fraud (generated)", np.log10(synthetic["Amount"] + 1), synth_fraud_color),
    ]
    for label, values, color in groups:
        ax.hist(values, bins=bins, density=True, histtype="step", lw=2, color=color,
                label=f"{label} (n={len(values):,})")
    ax.set_xlabel("log10(Amount + 1)   (1 = 9, 2 = 99, 3 = 999)")
    ax.set_ylabel("Density (each group scaled to equal area)")
    ax.set_title("SYNTHETIC DATASET - transaction amount distribution (not real-world data)", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(AMOUNT_PNG, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    check_outputs_absent()
    original_sha256 = sha256_of(ORIGINAL_PATH)

    print("===== 1. LOADING ORIGINAL DATASET (read only) =====")
    original = load_original()
    if list(original.columns) != COLUMNS:
        raise RuntimeError(f"Unexpected columns in {ORIGINAL_PATH}")
    counts = target_counts(original)
    for key, value in counts.items():
        print(f"  {key:<24} {value:,.5f}" if isinstance(value, float) else f"  {key:<24} {value:,}")

    print("\n===== 2. GENERATING SYNTHETIC FRAUD ROWS =====")
    synthetic, source_index, parameters = generate_synthetic_fraud(original, counts["synthetic_fraud"])
    print(f"  generated {len(synthetic):,} rows from {len(np.unique(source_index))} distinct source fraud rows")
    print(f"  Time values clipped to the original range: {parameters['time_values_clipped']}")

    print("\n===== 3. VALIDATING IN MEMORY =====")
    combined = pd.concat([original, synthetic], ignore_index=True)
    checks, info = validate_synthetic(combined, original, expected=counts)
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Validation failed before writing anything: {failed}")
    print(f"  {len(checks)} checks passed")

    print("\n===== 4. WRITING CSV (temporary file, re-read, validate, rename) =====")
    check_outputs_absent()
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    temporary = write_synthetic_csv(ORIGINAL_PATH, synthetic, SYNTHETIC_PATH)
    reread = pd.read_csv(temporary, float_precision="round_trip")
    checks, info = validate_synthetic(reread, original, synthetic_file=temporary,
                                      original_sha256=original_sha256, expected=counts)
    checks["file_round_trip_equals_generated_values"] = bool(reread.iloc[len(original):].reset_index(drop=True).equals(
        synthetic.reset_index(drop=True)))
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        temporary.unlink()  # remove only the temporary file this run created
        raise RuntimeError(f"Validation of the written file failed: {failed}")
    temporary.rename(SYNTHETIC_PATH)
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    for name, value in info.items():
        print(f"  info: {name} = {value}")

    print("\n===== 5. METADATA =====")
    metadata = {
        "dataset": "data/synthetic/creditcard_synthetic_5pct.csv",
        "description": (
            "Synthetic version of the Kaggle credit card fraud dataset with a 5% fraud rate. Rows 0-"
            f"{counts['original_rows'] - 1} are ALL original transactions, unchanged and in the original order. "
            f"Rows {counts['original_rows']}-{counts['final_rows'] - 1} are synthetic fraudulent transactions."
        ),
        "artificial_data_statement": (
            "The synthetic transactions are ARTIFICIAL. They are random numerical variations of real "
            "fraudulent rows and are NOT real-world fraud records. V1-V28 are anonymised PCA components, so "
            "the variations do not represent any specific real-world fraud technique. Use only for additional "
            "experiments and demonstrations; never for the official baseline or Improved v2 evaluation."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator_script": "src/generate_synthetic_data.py",
        "original_dataset": {"path": "data/creditcard.csv", "sha256": original_sha256},
        "counts": counts,
        "target_rule": "Keep all original rows; final_fraud = round(0.05 * legitimate / 0.95)",
        "random_seed": SEED,
        "source_selection": (
            "Each synthetic row has one source row drawn uniformly at random WITH replacement from the "
            "492 original rows where Class == 1 (numpy default_rng(42).integers)."
        ),
        "perturbation_formula": {
            "V1..V28": "x_new = x_source + epsilon * sigma_j * z_j",
            "Amount": "a_new = round(a_source * exp(epsilon * z_a), 2)",
            "Time": "t_new = clip(round(t_source + epsilon * sigma_time * z_t), time_min, time_max)",
            "Class": "1 (copied label; never used to compute the variation)",
            "z": "independent standard normal draws, one per value",
        },
        "perturbation_parameters": parameters,
        "synthetic_rows": {
            "first_row_position": counts["original_rows"],
            "last_row_position": counts["final_rows"] - 1,
            "distinct_source_rows_used": int(len(np.unique(source_index))),
            "source_original_row_index": [int(i) for i in source_index],
        },
        "output_file_sha256": sha256_of(SYNTHETIC_PATH),
        "validation": {"checks": checks, "all_passed": all(checks.values()), "info": info},
        "leakage_warning": (
            "Synthetic rows are derived from ALL 492 original fraud rows, including the 98 in the official "
            "test set (models/test_indices.csv), and the original legitimate rows include the rows both "
            "models were trained on. Do not evaluate the existing models on this file as if it were unseen data."
        ),
    }
    with open(METADATA_PATH, "x") as file:
        json.dump(metadata, file, indent=2)
    print(f"  saved {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    print("\n===== 6. CHARTS =====")
    check_outputs_absent([CLASS_PNG, AMOUNT_PNG])
    make_charts(original, synthetic, counts)
    print(f"  saved {CLASS_PNG.relative_to(PROJECT_ROOT)} and {AMOUNT_PNG.relative_to(PROJECT_ROOT)}")

    if sha256_of(ORIGINAL_PATH) != original_sha256:
        raise RuntimeError("data/creditcard.csv changed during the run!")
    print("\nOriginal dataset unchanged. ===== DONE =====")


if __name__ == "__main__":
    main()
