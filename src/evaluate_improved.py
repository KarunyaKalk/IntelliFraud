"""
Evaluate the "Improved v2" Autoencoder and compare it with the original
baseline on EXACTLY the same official test rows.

* Improved v2 is loaded from models/improved/ (read only).
* Baseline scores come from the committed results/metrics/baseline_test_scores.csv
  (read only) - the original model is not re-run, so its results cannot change.
* Metrics, stratified bootstrap and fingerprints use the functions of
  src/baseline_evaluation.py (imported, not modified).
* Paired bootstrap: 1,000 stratified resamples (seed 42). In every resample BOTH
  models are scored on the same rows; the Average Precision difference
  (Improved v2 - baseline) is recorded. Improvement is called statistically
  supported ONLY if the whole 95% CI of that difference is above zero.

Outputs (new files only; the script stops if any already exists):
  results/metrics/improved_evaluation_report.txt
  results/metrics/improved_metrics.csv
  results/metrics/improved_test_scores.csv
  results/metrics/model_comparison_baseline_vs_improved.csv
  results/graphs/improved_training_history.png
  results/graphs/improved_pr_curve.png
  results/graphs/improved_reconstruction_error_distribution.png
  results/graphs/improved_confusion_matrix.png
  results/graphs/pr_curve_baseline_vs_improved.png

Run from the project folder:
    python src/evaluate_improved.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

from src.anomaly_detection import detect_anomalies, load_model_components
from src.baseline_evaluation import (  # imported, not modified
    BOOTSTRAP_SEED,
    CI_LEVEL,
    N_BOOTSTRAP,
    Report,
    bootstrap_ci,
    compute_metrics,
    sha256_of,
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
TEST_INDEX_PATH = PROJECT_ROOT / "models" / "test_indices.csv"
IMPROVED_DIR = PROJECT_ROOT / "models" / "improved"
BASELINE_SCORES_PATH = PROJECT_ROOT / "results" / "metrics" / "baseline_test_scores.csv"
BASELINE_METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "baseline_metrics.csv"

METRIC_DIR = PROJECT_ROOT / "results" / "metrics"
GRAPH_DIR = PROJECT_ROOT / "results" / "graphs"
REPORT_PATH = METRIC_DIR / "improved_evaluation_report.txt"
METRICS_CSV_PATH = METRIC_DIR / "improved_metrics.csv"
SCORES_CSV_PATH = METRIC_DIR / "improved_test_scores.csv"
COMPARISON_CSV_PATH = METRIC_DIR / "model_comparison_baseline_vs_improved.csv"
HISTORY_PNG = GRAPH_DIR / "improved_training_history.png"
PR_PNG = GRAPH_DIR / "improved_pr_curve.png"
ERROR_PNG = GRAPH_DIR / "improved_reconstruction_error_distribution.png"
CONFUSION_PNG = GRAPH_DIR / "improved_confusion_matrix.png"
PR_COMPARE_PNG = GRAPH_DIR / "pr_curve_baseline_vs_improved.png"
OUTPUTS = [REPORT_PATH, METRICS_CSV_PATH, SCORES_CSV_PATH, COMPARISON_CSV_PATH,
           HISTORY_PNG, PR_PNG, ERROR_PNG, CONFUSION_PNG, PR_COMPARE_PNG]

# Files that must be byte-identical before and after this script.
PROTECTED_FILES = [
    DATA_PATH,
    PROJECT_ROOT / "models" / "autoencoder.keras",
    PROJECT_ROOT / "models" / "scaler.pkl",
    PROJECT_ROOT / "models" / "threshold.txt",
    TEST_INDEX_PATH,
    PROJECT_ROOT / "models" / "training_history.csv",
    PROJECT_ROOT / "models" / "evaluation_results.csv",
    PROJECT_ROOT / "models" / "model_comparison.csv",
    BASELINE_SCORES_PATH,
    BASELINE_METRICS_PATH,
    PROJECT_ROOT / "results" / "metrics" / "baseline_evaluation_report.txt",
    *(IMPROVED_DIR / name for name in [
        "autoencoder.keras", "scaler.pkl", "threshold.txt", "training_history.csv",
        "config.json", "train_indices.csv", "validation_indices.csv"]),
]

BASELINE_COLOR = "#4C72B0"
IMPROVED_COLOR = "#2E8B57"
FRAUD_COLOR = "#D1495B"
NEUTRAL_COLOR = "#555555"

PAIRED_METRICS = ["average_precision", "pr_auc_trapezoidal", "roc_auc", "precision", "recall", "f1_score"]


class OverwriteError(RuntimeError):
    """Raised instead of overwriting existing files."""


def shown(path):
    """Display name like 'results/metrics/file.csv' (works for any output folder)."""
    return f"{path.parent.parent.name}/{path.parent.name}/{path.name}"


def check_outputs_absent(paths=OUTPUTS):
    existing = [shown(p) for p in paths if p.exists()]
    if existing:
        raise OverwriteError(
            "Refusing to run because these output files already exist: "
            + ", ".join(existing) + ". Move or delete them yourself first."
        )


def write_guard(path):
    """Final check immediately before writing one file."""
    if path.exists():
        raise OverwriteError(f"{path} appeared during the run - not overwriting it.")
    return path


def _point_metrics(y, scores, preds):
    tp = int(((y == 1) & (preds == 1)).sum())
    fp = int(((y == 0) & (preds == 1)).sum())
    fn = int(((y == 1) & (preds == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    cp, cr, _ = precision_recall_curve(y, scores)
    return {
        "average_precision": average_precision_score(y, scores),
        "pr_auc_trapezoidal": auc(cr, cp),
        "roc_auc": roc_auc_score(y, scores),
        "precision": p,
        "recall": r,
        "f1_score": 2 * p * r / (p + r) if (p + r) else 0.0,
    }


def paired_bootstrap(y_true, base_scores, base_pred, new_scores, new_pred):
    """
    Stratified PAIRED bootstrap (same procedure and seed as the baseline CIs).

    Each resample draws legitimate rows from legitimate rows and fraud rows from
    fraud rows (class counts preserved). Both models are evaluated on the SAME
    resampled rows and the difference (improved - baseline) is stored.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    y_true = np.asarray(y_true)
    normal_rows = np.flatnonzero(y_true == 0)
    fraud_rows = np.flatnonzero(y_true == 1)
    differences = {name: [] for name in PAIRED_METRICS}
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate([
            rng.choice(normal_rows, size=len(normal_rows), replace=True),
            rng.choice(fraud_rows, size=len(fraud_rows), replace=True),
        ])
        yt = y_true[idx]
        base = _point_metrics(yt, base_scores[idx], base_pred[idx])
        new = _point_metrics(yt, new_scores[idx], new_pred[idx])
        for name in PAIRED_METRICS:
            differences[name].append(new[name] - base[name])
    alpha = (100 - CI_LEVEL) / 2
    return {
        name: {
            "low": float(np.percentile(vals, alpha)),
            "high": float(np.percentile(vals, 100 - alpha)),
            "share_above_zero": float(np.mean(np.asarray(vals) > 0)),
        }
        for name, vals in differences.items()
    }


def main():
    check_outputs_absent()
    report = Report()
    report.add("IntelliFraud - Improved v2 Autoencoder: evaluation and paired comparison with the baseline")
    report.add("(read-only for both models; no training, no threshold tuning on test data)")

    hashes_before = {p: sha256_of(p) for p in PROTECTED_FILES}
    report.section("0. INPUT FILES (SHA-256 fingerprints)")
    for path, digest in hashes_before.items():
        report.add(f"{str(path.relative_to(PROJECT_ROOT)):<42} {digest}")

    config = json.loads((IMPROVED_DIR / "config.json").read_text())

    # ==============================================================
    # 1. Official test rows - identical to the baseline
    # ==============================================================
    df = pd.read_csv(DATA_PATH)
    test_index = pd.Index(pd.read_csv(TEST_INDEX_PATH)["index"].astype(int))
    baseline_scores = pd.read_csv(BASELINE_SCORES_PATH, float_precision="round_trip")

    report.section("1. VERIFICATION - same official test rows as the baseline")
    same_rows = np.array_equal(baseline_scores["test_row_index"].to_numpy(), test_index.to_numpy())
    report.add(f"Official test rows (models/test_indices.csv): {len(test_index):,}")
    report.add(f"Baseline scores cover the same rows in the same order: {same_rows}")
    if not same_rows:
        raise RuntimeError("baseline_test_scores.csv rows differ from models/test_indices.csv - stopping.")

    test_df = df.loc[test_index]
    y_true = test_df["Class"].to_numpy()
    labels_match = np.array_equal(y_true, baseline_scores["actual_class"].to_numpy())
    report.add(f"Class labels identical to the baseline file: {labels_match}")
    if not labels_match:
        raise RuntimeError("Class labels differ from the baseline scores file - stopping.")

    base_scores = baseline_scores["reconstruction_error"].to_numpy()
    base_pred = baseline_scores["predicted_class"].to_numpy()
    base_threshold = float((PROJECT_ROOT / "models" / "threshold.txt").read_text())
    baseline = compute_metrics(y_true, base_scores, base_pred)
    stored = pd.read_csv(BASELINE_METRICS_PATH)
    stored = stored[stored.evaluation_set == "official_full_test"].set_index("metric")["value"]
    baseline_consistent = all(
        np.isclose(baseline[k], stored[k]) for k in
        ["true_negatives", "false_positives", "false_negatives", "true_positives", "average_precision"]
    )
    report.add(f"Baseline metrics recomputed from its scores match baseline_metrics.csv: {baseline_consistent} "
               f"(TN {baseline['true_negatives']:,}, FP {baseline['false_positives']:,}, "
               f"FN {baseline['false_negatives']:,}, TP {baseline['true_positives']:,})")
    if not baseline_consistent:
        raise RuntimeError("Baseline scores and stored baseline metrics disagree - stopping.")

    # ==============================================================
    # 2. Score Improved v2 on the same rows
    # ==============================================================
    components = load_model_components(
        IMPROVED_DIR / "autoencoder.keras", IMPROVED_DIR / "scaler.pkl", IMPROVED_DIR / "threshold.txt"
    )
    new_threshold = components[2]
    results = detect_anomalies(test_df, components)  # same reconstruction error as the baseline
    new_scores = results["reconstruction_error"].to_numpy()
    new_pred = results["anomaly"].to_numpy()
    improved = compute_metrics(y_true, new_scores, new_pred)

    report.section("2. IMPROVED v2 SET-UP (from models/improved/config.json)")
    counts = config["row_counts"]
    for key in ["training_split_rows", "training_split_fraud_rows_excluded", "legitimate_training_pool_rows",
                "repeated_copies_removed", "after_removing_repeats", "test_set_twins_removed",
                "clean_pool_rows", "training_rows", "validation_rows"]:
        report.add(f"{key:<38} {counts[key]:>8,}")
    r = config["results"]
    report.add(f"Epochs completed: {r['epochs_completed']} (max {config['training_parameters']['max_epochs']}), "
               f"best epoch {r['best_epoch']}, stopped early: {r['stopped_early']}, "
               f"training time {r['training_seconds']} s")
    report.add(f"Threshold: {new_threshold:.6f} = {config['threshold_rule']} (baseline: {base_threshold:.6f}, "
               f"95th percentile of its own training errors)")
    report.add(f"Leakage checks passed during training: {all(config['leakage_checks'].values())}")

    # ==============================================================
    # 3. Metrics with stratified bootstrap CIs (same function as baseline)
    # ==============================================================
    print(f"\nBootstrap CIs ({N_BOOTSTRAP} stratified resamples, seed {BOOTSTRAP_SEED})...")
    base_ci = bootstrap_ci(y_true, base_scores, base_pred)
    new_ci = bootstrap_ci(y_true, new_scores, new_pred)
    print("Paired bootstrap...")
    paired = paired_bootstrap(y_true, base_scores, base_pred, new_scores, new_pred)

    def pct(x):
        return f"{x * 100:.2f}%"

    def fmt(name, value):
        return f"{value:.4f}" if name in ("average_precision", "pr_auc_trapezoidal", "roc_auc") else pct(value)

    def ci_txt(ci, name):
        low, high = ci[name]
        return f"[{fmt(name, low)} - {fmt(name, high)}]"

    report.section("3. OFFICIAL TEST SET: BASELINE vs IMPROVED v2 (56,962 identical rows)")
    report.add(f"{'Metric':<24}{'Baseline':>10} {'95% CI':<20}{'Improved v2':>12} {'95% CI':<20}")
    for label, name in [("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1_score"),
                        ("Average Precision", "average_precision"), ("PR-AUC (trapezoidal)", "pr_auc_trapezoidal"),
                        ("ROC-AUC", "roc_auc")]:
        report.add(f"{label:<24}{fmt(name, baseline[name]):>10} {ci_txt(base_ci, name):<20}"
                   f"{fmt(name, improved[name]):>12} {ci_txt(new_ci, name):<20}")
    for label, name in [("Specificity", "specificity"), ("False-positive rate", "false_positive_rate"),
                        ("Accuracy", "accuracy")]:
        report.add(f"{label:<24}{pct(baseline[name]):>10} {'':<20}{pct(improved[name]):>12}")
    report.add(f"{'False alarms per catch':<24}{baseline['false_alarms_per_fraud_caught']:>10.1f} {'':<20}"
               f"{improved['false_alarms_per_fraud_caught']:>12.1f}")
    report.add()
    report.add("Confusion matrices (rows = actual, columns = predicted normal / anomaly):")
    for name, m in [("Baseline", baseline), ("Improved v2", improved)]:
        report.add(f"  {name:<12} legit: {m['true_negatives']:>6,} / {m['false_positives']:>5,}    "
                   f"fraud: {m['false_negatives']:>3} / {m['true_positives']:>3}")
    report.add()
    report.add("CI method: stratified percentile bootstrap, identical to the baseline report "
               f"({N_BOOTSTRAP:,} resamples, seed {BOOTSTRAP_SEED}, class counts preserved, thresholds fixed).")

    # ==============================================================
    # 4. Paired bootstrap decision (pre-registered rule)
    # ==============================================================
    ap_diff = improved["average_precision"] - baseline["average_precision"]
    ap = paired["average_precision"]
    supported = ap["low"] > 0

    report.section("4. PAIRED BOOTSTRAP - is Improved v2 better? (primary metric: Average Precision)")
    report.add("Method: in each of 1,000 stratified resamples (seed 42) BOTH models are scored on the same")
    report.add("resampled test rows; the difference Improved v2 - baseline is recorded. 95% CI = 2.5th-97.5th")
    report.add("percentile. Pre-agreed rule: improvement is statistically supported ONLY if the whole CI > 0.")
    report.add()
    report.add(f"Observed AP: baseline {baseline['average_precision']:.4f}, Improved v2 "
               f"{improved['average_precision']:.4f}")
    report.add(f"Observed AP difference (Improved v2 - baseline): {ap_diff:+.4f}")
    report.add(f"Paired 95% CI for the difference: [{ap['low']:+.4f}, {ap['high']:+.4f}]")
    report.add(f"Share of resamples where Improved v2 had higher AP: {ap['share_above_zero']:.1%}")
    report.add()
    if supported:
        report.add("DECISION: Improvement in Average Precision IS statistically supported "
                   "(entire 95% CI above zero).")
    elif ap["high"] < 0:
        report.add("DECISION: Improvement NOT established. The entire 95% CI is BELOW zero: Improved v2 has "
                   "lower Average Precision than the baseline.")
    else:
        report.add("DECISION: Improvement NOT established. The 95% CI includes zero, so the data cannot "
                   "distinguish the two models on Average Precision.")
    report.add()
    report.add("Secondary paired differences (descriptive only, not used for the decision):")
    for label, name in [("PR-AUC (trapezoidal)", "pr_auc_trapezoidal"), ("ROC-AUC", "roc_auc"),
                        ("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1_score")]:
        diff = improved[name] - baseline[name]
        p = paired[name]
        scale = 1 if name in ("pr_auc_trapezoidal", "roc_auc") else 100
        unit = "" if scale == 1 else " pp"
        report.add(f"  {label:<22} {diff * scale:+.4f}{unit}   95% CI [{p['low'] * scale:+.4f}, "
                   f"{p['high'] * scale:+.4f}]{unit}")

    # ==============================================================
    # 5. Sensitivity analysis (same 344 rows as the baseline report)
    # ==============================================================
    keep = ~baseline_scores["duplicate_overlap"].to_numpy().astype(bool)
    base_s = compute_metrics(y_true[keep], base_scores[keep], base_pred[keep])
    new_s = compute_metrics(y_true[keep], new_scores[keep], new_pred[keep])
    report.section("5. SENSITIVITY ANALYSIS (NOT official) - same 344 duplicate-overlap rows removed")
    report.add(f"Rows kept: {int(keep.sum()):,} (fraud {int(y_true[keep].sum())}); the removed rows are those "
               f"with an exact copy in the ORIGINAL baseline training split.")
    report.add(f"{'Metric':<24}{'Baseline':>10}{'Improved v2':>14}")
    for label, name in [("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1_score"),
                        ("Average Precision", "average_precision"), ("PR-AUC (trapezoidal)", "pr_auc_trapezoidal"),
                        ("ROC-AUC", "roc_auc")]:
        report.add(f"{label:<24}{fmt(name, base_s[name]):>10}{fmt(name, new_s[name]):>14}")
    report.add("Note: Improved v2 has NO exact overlap with the test set by construction (test twins removed).")

    # ==============================================================
    # 6. Save tables
    # ==============================================================
    rows = []
    for set_name, metrics, ci in [("official_full_test", improved, new_ci),
                                  ("sensitivity_duplicates_excluded", new_s, None)]:
        for key, val in metrics.items():
            low, high = (ci or {}).get(key, (np.nan, np.nan))
            rows.append({"evaluation_set": set_name, "metric": key, "value": val, "ci95_low": low, "ci95_high": high})
    rows.append({"evaluation_set": "official_full_test", "metric": "threshold", "value": new_threshold,
                 "ci95_low": np.nan, "ci95_high": np.nan})
    pd.DataFrame(rows).to_csv(write_guard(METRICS_CSV_PATH), index=False)

    pd.DataFrame({
        "test_row_index": test_index.to_numpy(),
        "actual_class": y_true.astype(int),
        "reconstruction_error": new_scores,
        "predicted_class": new_pred.astype(int),
        "duplicate_overlap": baseline_scores["duplicate_overlap"].to_numpy(),
    }).to_csv(write_guard(SCORES_CSV_PATH), index=False)

    comparison_rows = []
    for key in ["precision", "recall", "f1_score", "average_precision", "pr_auc_trapezoidal", "roc_auc",
                "specificity", "false_positive_rate", "accuracy", "true_negatives", "false_positives",
                "false_negatives", "true_positives", "false_alarms_per_fraud_caught"]:
        p = paired.get(key)
        comparison_rows.append({
            "metric": key, "baseline": baseline[key], "improved_v2": improved[key],
            "difference_improved_minus_baseline": improved[key] - baseline[key],
            "paired_ci95_low": p["low"] if p else np.nan, "paired_ci95_high": p["high"] if p else np.nan,
            "primary_metric": key == "average_precision",
        })
    comparison_rows.append({"metric": "threshold", "baseline": base_threshold, "improved_v2": new_threshold,
                            "difference_improved_minus_baseline": new_threshold - base_threshold,
                            "paired_ci95_low": np.nan, "paired_ci95_high": np.nan, "primary_metric": False})
    comparison_rows.append({"metric": "improvement_statistically_supported", "baseline": np.nan,
                            "improved_v2": np.nan, "difference_improved_minus_baseline": float(supported),
                            "paired_ci95_low": np.nan, "paired_ci95_high": np.nan, "primary_metric": True})
    pd.DataFrame(comparison_rows).to_csv(write_guard(COMPARISON_CSV_PATH), index=False)

    # ==============================================================
    # 7. Graphs
    # ==============================================================
    history = pd.read_csv(IMPROVED_DIR / "training_history.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["loss"], color=BASELINE_COLOR, lw=2, label="Training loss")
    ax.plot(history["epoch"], history["val_loss"], color=IMPROVED_COLOR, lw=2, label="Validation loss")
    best = int(history["val_loss"].idxmin())
    ax.scatter([history["epoch"][best]], [history["val_loss"][best]], color=FRAUD_COLOR, zorder=5,
               label=f"Best epoch {history['epoch'][best]} (weights kept)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error")
    ax.set_title(f"Improved v2 Training and Validation Loss ({len(history)} Epochs)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(write_guard(HISTORY_PNG), dpi=150)
    plt.close(fig)

    def pr_plot(path, curves, title):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for name, scores, metrics, color in curves:
            cp, cr, _ = precision_recall_curve(y_true, scores)
            ax.plot(cr, cp, color=color, lw=2,
                    label=f"{name} (AP {metrics['average_precision']:.4f}, PR-AUC {metrics['pr_auc_trapezoidal']:.4f})")
            ax.scatter([metrics["recall"]], [metrics["precision"]], color=color, edgecolor="black", zorder=5, s=55,
                       label=f"{name} threshold (recall {pct(metrics['recall'])}, precision {pct(metrics['precision'])})")
        ax.axhline(y_true.mean(), color=NEUTRAL_COLOR, ls="--", lw=1, label=f"Random guessing ({y_true.mean():.4f})")
        ax.set_xlabel("Recall (share of fraud caught)")
        ax.set_ylabel("Precision (share of alerts that are fraud)")
        ax.set_title(title)
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(write_guard(path), dpi=150)
        plt.close(fig)

    pr_plot(PR_PNG, [("Improved v2", new_scores, improved, IMPROVED_COLOR)],
            "Improved v2 - Precision-Recall curve (official test set)")
    pr_plot(PR_COMPARE_PNG, [("Baseline", base_scores, baseline, BASELINE_COLOR),
                             ("Improved v2", new_scores, improved, IMPROVED_COLOR)],
            "Baseline vs Improved v2 - same 56,962 test rows")

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(np.log10(new_scores.min()), np.log10(new_scores.max()), 70)
    ax.hist(new_scores[y_true == 0], bins=bins, density=True, alpha=0.55, color=BASELINE_COLOR,
            label=f"Legitimate (n={int((y_true == 0).sum()):,})")
    ax.hist(new_scores[y_true == 1], bins=bins, density=True, alpha=0.6, color=FRAUD_COLOR,
            label=f"Fraud (n={int(y_true.sum())})")
    ax.axvline(new_threshold, color="black", ls="--", lw=1.2, label=f"Threshold = {new_threshold:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction error (log scale)")
    ax.set_ylabel("Density (each class scaled to equal area)")
    ax.set_title("Improved v2 - reconstruction error on the official test set")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(write_guard(ERROR_PNG), dpi=150)
    plt.close(fig)

    cm = np.array([[improved["true_negatives"], improved["false_positives"]],
                   [improved["false_negatives"], improved["true_positives"]]])
    share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.imshow(share, cmap="Greens", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({share[i, j]:.2%} of row)", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["Predicted normal", "Predicted anomaly"])
    ax.set_yticks([0, 1], ["Actual legitimate", "Actual fraud"])
    ax.set_title(f"Improved v2 confusion matrix (threshold {new_threshold:.4f})")
    fig.tight_layout()
    fig.savefig(write_guard(CONFUSION_PNG), dpi=150)
    plt.close(fig)

    # ==============================================================
    # 8. Limitations and integrity
    # ==============================================================
    report.section("6. LIMITATIONS")
    max_epochs = config["training_parameters"]["max_epochs"]
    epochs_done = int(r["epochs_completed"])
    best_epoch = int(r["best_epoch"])
    if not r["stopped_early"] and epochs_done == max_epochs:
        window = history.tail(8)
        training_note = (
            f"Training reached the maximum of {max_epochs} epochs and early stopping did not trigger. "
            f"The best validation loss ({r['best_val_loss']:.4f}) occurred at epoch {best_epoch}"
            + (", the final epoch" if best_epoch == epochs_done else "")
            + f". Validation loss was still decreasing slowly near the end of training "
            f"({window['val_loss'].iloc[0]:.4f} at epoch {int(window['epoch'].iloc[0])} -> "
            f"{window['val_loss'].iloc[-1]:.4f} at epoch {int(window['epoch'].iloc[-1])}), so the model "
            f"may not be fully converged. Further training might be worth exploring in future work; "
            f"no additional training was performed for this evaluation."
        )
    else:
        training_note = (
            f"Early stopping ended training after {epochs_done} of {max_epochs} epochs; the best validation "
            f"loss ({r['best_val_loss']:.4f}) occurred at epoch {best_epoch} and those weights were kept."
        )
    for i, text in enumerate([
        training_note,
        "Only 98 test frauds: confidence intervals are wide and small differences are not meaningful.",
        "Same random (not time-based) split and the same Time feature as the baseline, so the "
        "time-ordering limitation still applies.",
        "The threshold rule (95th percentile of legitimate validation errors) fixes the false-alarm "
        "rate on legitimate data near 5%; it is not optimised for fraud detection (by design: no labels).",
        "Reproducibility: seeds and TensorFlow op determinism were set; identical results are expected on "
        "the same machine and software versions, not guaranteed across different hardware.",
        "The 70 duplicate copies inside the official test set were kept, so both models are compared on "
        "the original test rows.",
    ], 1):
        report.add(f"{i}. {text}")

    report.section("7. INTEGRITY CHECK AND OUTPUT FILES")
    unchanged = all(sha256_of(p) == h for p, h in hashes_before.items())
    report.add(f"Baseline, dataset and Improved v2 model files unchanged after run: {unchanged}")
    for path in OUTPUTS:
        report.add(shown(path))
    report.save(write_guard(REPORT_PATH))
    if not unchanged:
        raise RuntimeError("A protected file changed during the run!")


if __name__ == "__main__":
    main()
