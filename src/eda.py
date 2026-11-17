"""
Exploratory Data Analysis (EDA) for the IntelliFraud project.

What this script does
---------------------
* Reads data/creditcard.csv (READ-ONLY - the file is never modified).
* Prints and saves basic facts about the dataset: size, columns, data types,
  class balance, missing values, duplicate rows and transaction amounts.
* Saves graphs to results/graphs/ and a text report to
  results/metrics/dataset_report.txt.

It does NOT remove duplicates, change the data, or train any model.

How to run (from the project root, with the virtual environment active):
    python src/eda.py
"""

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to image files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Paths (based on this file's location, so it works from any folder)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
TEST_INDEX_PATH = PROJECT_ROOT / "models" / "test_indices.csv"
GRAPH_DIR = PROJECT_ROOT / "results" / "graphs"
METRIC_DIR = PROJECT_ROOT / "results" / "metrics"
REPORT_PATH = METRIC_DIR / "dataset_report.txt"

LEGIT_COLOR = "#4C72B0"
FRAUD_COLOR = "#D1495B"

# Show every column when printing tables
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


class Report:
    """Collects text so it is both printed and saved to the report file."""

    def __init__(self):
        self.lines = []

    def add(self, text=""):
        text = str(text)
        print(text)
        self.lines.append(text)

    def section(self, title):
        self.add()
        self.add("=" * 70)
        self.add(title)
        self.add("=" * 70)

    def save(self, path):
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def sha256_of(path):
    """Fingerprint of a file, so the report records exactly which data was used."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_bars(ax, bars, values, total):
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:,}\n({value / total:.3%})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9,
        )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    report = Report()
    report.add("IntelliFraud - Dataset EDA Report")
    report.add(f"Dataset file : data/creditcard.csv")
    report.add(f"SHA-256      : {sha256_of(DATA_PATH)}")
    report.add(f"pandas {pd.__version__}, numpy {np.__version__}, matplotlib {matplotlib.__version__}")

    # 1. Load -------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    # 2. Dimensions and first rows ---------------------------------
    report.section("1. DATASET DIMENSIONS")
    report.add(f"Rows (transactions): {df.shape[0]:,}")
    report.add(f"Columns            : {df.shape[1]}")
    report.add(f"Memory in pandas   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    report.section("2. FIRST FIVE ROWS (shown sideways: one column per row)")
    report.add(df.head(5).T.to_string())

    # 3. Columns and dtypes ----------------------------------------
    report.section("3. COLUMN NAMES AND DATA TYPES")
    report.add(df.dtypes.to_string())
    report.add()
    report.add("Data type counts: " + ", ".join(
        f"{dtype}={count}" for dtype, count in df.dtypes.astype(str).value_counts().items()
    ))

    # 4-5. Class balance -------------------------------------------
    report.section("4. CLASS BALANCE (Class: 0 = legitimate, 1 = fraud)")
    counts = df["Class"].value_counts().sort_index()
    total = len(df)
    legit, fraud = int(counts.get(0, 0)), int(counts.get(1, 0))
    report.add(f"Legitimate transactions: {legit:,}")
    report.add(f"Fraudulent transactions: {fraud:,}")
    report.add(f"Fraud percentage       : {fraud / total * 100:.4f}%")
    report.add(f"Ratio                  : about 1 fraud per {legit / fraud:,.0f} legitimate transactions")
    report.add(f"Accuracy of a model that ALWAYS says 'legitimate': {legit / total * 100:.4f}%")

    # 6. Missing values --------------------------------------------
    report.section("5. MISSING VALUES PER COLUMN")
    missing = df.isnull().sum()
    report.add(missing.to_string())
    report.add()
    report.add(f"Total missing values: {int(missing.sum()):,}")

    # 7. Duplicates (counted only - NOT removed) --------------------
    report.section("6. DUPLICATE ROWS (counted only, NOT removed)")
    dup_extra = df.duplicated(keep="first")
    dup_all = df.duplicated(keep=False)
    report.add(f"Extra copies (rows identical to an earlier row): {int(dup_extra.sum()):,}")
    report.add(f"Rows involved in duplication (incl. originals) : {int(dup_all.sum()):,}")
    report.add(f"Distinct rows after hypothetical de-duplication: {total - int(dup_extra.sum()):,}")
    report.add(f"Extra copies that are legitimate (Class 0)     : {int((dup_extra & (df['Class'] == 0)).sum()):,}")
    report.add(f"Extra copies that are fraud (Class 1)          : {int((dup_extra & (df['Class'] == 1)).sum()):,}")
    report.add(f"Fraud rows involved in duplication             : {int((dup_all & (df['Class'] == 1)).sum()):,}")
    feature_cols = [c for c in df.columns if c != "Class"]
    conflicting = df[df.duplicated(subset=feature_cols, keep=False)].groupby(feature_cols)["Class"].nunique()
    report.add(f"Identical features but DIFFERENT labels        : {int((conflicting > 1).sum()):,} groups")

    # Read-only check against the existing train/test split (models/test_indices.csv)
    if TEST_INDEX_PATH.exists():
        test_idx = pd.Index(pd.read_csv(TEST_INDEX_PATH)["index"].astype(int))
        is_test = df.index.isin(test_idx)
        row_keys = pd.util.hash_pandas_object(df, index=False)
        train_keys = set(row_keys[~is_test])
        test_in_train = row_keys[is_test].isin(train_keys)
        report.add()
        report.add("Duplicates across the EXISTING train/test split (models/test_indices.csv, read-only):")
        report.add(f"  Test rows                                     : {int(is_test.sum()):,}")
        report.add(f"  Test rows with an exact copy in training data : {int(test_in_train.sum()):,}")
        report.add(f"  ...of which are fraud                         : "
                   f"{int((test_in_train & (df.loc[is_test, 'Class'] == 1)).sum()):,}")

    # 8. Amount statistics -----------------------------------------
    report.section("7. TRANSACTION AMOUNT STATISTICS (Amount column)")
    amount_stats = pd.DataFrame({
        "All": df["Amount"].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]),
        "Legitimate": df.loc[df["Class"] == 0, "Amount"].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]),
        "Fraud": df.loc[df["Class"] == 1, "Amount"].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]),
    })
    report.add(amount_stats.round(2).to_string())
    report.add()
    for name, mask in [("All", slice(None)), ("Legitimate", df["Class"] == 0), ("Fraud", df["Class"] == 1)]:
        amounts = df.loc[mask, "Amount"]
        report.add(f"{name:<11} skewness={amounts.skew():7.2f} | zero-amount transactions={int((amounts == 0).sum()):,} "
                   f"({(amounts == 0).mean():.2%}) | sum={amounts.sum():,.2f}")

    report.section("8. TIME COLUMN (seconds since the first transaction)")
    report.add(f"Min: {df['Time'].min():,.0f} s | Max: {df['Time'].max():,.0f} s "
               f"(= {df['Time'].max() / 3600:.1f} hours)")

    # 9. Class distribution graph ----------------------------------
    graphs = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, log in zip(axes, [False, True]):
        bars = ax.bar(["Legitimate (0)", "Fraud (1)"], [legit, fraud], color=[LEGIT_COLOR, FRAUD_COLOR])
        label_bars(ax, bars, [legit, fraud], total)
        ax.set_ylabel("Number of transactions" + (" (log scale)" if log else ""))
        if log:
            ax.set_yscale("log")
            ax.set_title("Same data, logarithmic scale")
            ax.set_ylim(top=legit * 8)
        else:
            ax.set_title("Legitimate vs fraudulent transactions")
            ax.set_ylim(top=legit * 1.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = GRAPH_DIR / "class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    graphs.append(path)

    # 10a. Overall amount distribution -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    p99 = df["Amount"].quantile(0.99)
    axes[0].hist(df.loc[df["Amount"] <= p99, "Amount"], bins=60, color=LEGIT_COLOR)
    axes[0].set_title(f"Amount (up to 99th percentile = {p99:,.2f})")
    axes[0].set_xlabel("Amount")
    axes[0].set_ylabel("Number of transactions")
    axes[1].hist(np.log10(df["Amount"] + 1), bins=60, color=LEGIT_COLOR)
    axes[1].set_title("All amounts, log10(Amount + 1)")
    axes[1].set_xlabel("log10(Amount + 1)   (1 = 9, 2 = 99, 3 = 999)")
    axes[1].set_ylabel("Number of transactions")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = GRAPH_DIR / "amount_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    graphs.append(path)

    # 10b. Amount by class ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bins = np.linspace(0, np.log10(df["Amount"].max() + 1), 50)
    for cls, name, color in [(0, "Legitimate", LEGIT_COLOR), (1, "Fraud", FRAUD_COLOR)]:
        values = np.log10(df.loc[df["Class"] == cls, "Amount"] + 1)
        axes[0].hist(values, bins=bins, density=True, alpha=0.55, color=color, label=f"{name} (n={len(values):,})")
    axes[0].set_title("Amount by class (each class scaled to equal area)")
    axes[0].set_xlabel("log10(Amount + 1)")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    box = axes[1].boxplot(
        [df.loc[df["Class"] == 0, "Amount"] + 1, df.loc[df["Class"] == 1, "Amount"] + 1],
        tick_labels=["Legitimate (0)", "Fraud (1)"], patch_artist=True, showfliers=True,
        flierprops={"markersize": 2, "alpha": 0.3},
    )
    for patch, color in zip(box["boxes"], [LEGIT_COLOR, FRAUD_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_yscale("log")
    axes[1].set_title("Amount + 1 by class (log scale)")
    axes[1].set_ylabel("Amount + 1")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = GRAPH_DIR / "amount_by_class.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    graphs.append(path)

    report.section("9. SAVED FILES")
    for path in graphs:
        report.add(path.relative_to(PROJECT_ROOT))
    report.add(REPORT_PATH.relative_to(PROJECT_ROOT))

    report.save(REPORT_PATH)


if __name__ == "__main__":
    main()
