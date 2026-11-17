"""
IntelliFraud - Streamlit dashboard.

Run from the project folder:
    python -m streamlit run app.py

What the page does
------------------
1. Dataset          - preview any CSV (default: data/creditcard.csv).
2. Anomaly Detection - score EVERY row of the loaded file with the saved
                      Autoencoder. Flagged rows are ANOMALIES (unusual), not
                      confirmed fraud. These counts are not accuracy figures.
3. Baseline Evaluation - the stored, verified results of the original model
                      on 56,962 held-out test transactions. They never change
                      when another file is uploaded.
4. Model Comparison - stored results from src/compare_models.py.

Caching note: the model is loaded once per dashboard session
(st.cache_resource). If the files in models/ are replaced (e.g. after
retraining), restart the dashboard or use the menu's "Clear cache".
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.anomaly_detection import (
    InputDataError,
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

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
TEST_INDEX_PATH = PROJECT_ROOT / "models" / "test_indices.csv"
BASELINE_METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "baseline_metrics.csv"
BASELINE_REPORT_PATH = "results/metrics/baseline_evaluation_report.txt"
COMPARISON_PATH = PROJECT_ROOT / "models" / "model_comparison.csv"

NORMAL_COLOR = "#4C72B0"
ANOMALY_COLOR = "#D1495B"

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IntelliFraud - Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #ddd;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🔍 IntelliFraud: Credit Card Anomaly Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Unsupervised Autoencoder that flags unusual transactions for review'
    '</div>',
    unsafe_allow_html=True
)


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_resource(show_spinner="Loading the saved Autoencoder, scaler and threshold...")
def get_model_components():
    """Loaded once per dashboard session. Errors are NOT cached."""
    return load_model_components()


@st.cache_data(show_spinner="Reading data/creditcard.csv...", max_entries=1)
def read_default_dataset(path_text, identity):
    # `identity` (size + modification time) makes the cache refresh if the file changes.
    return pd.read_csv(path_text)


@st.cache_data(show_spinner="Reading the uploaded file...", max_entries=2)
def read_uploaded_dataset(identity, _uploaded_file):
    _uploaded_file.seek(0)
    return pd.read_csv(_uploaded_file)


@st.cache_data
def read_baseline_metrics(path_text):
    return load_baseline_metrics(path_text)


@st.cache_data
def read_comparison(path_text):
    return pd.read_csv(path_text)


@st.cache_data
def training_split_counts(test_index_path, identity, _df):
    """
    Counts for the default dataset, using the saved test split.

    identity (file size + modification time) is the cache key; _df is not hashed.
    Returns test rows, training-split rows, and how many training-split rows
    are legitimate (used to train the Autoencoder and set the threshold)
    or fraudulent (excluded from training).
    """
    test_index = pd.read_csv(test_index_path)["index"].astype(int)
    in_training_split = ~_df.index.isin(test_index)
    return {
        "test_rows": int((~in_training_split).sum()),
        "training_split_rows": int(in_training_split.sum()),
        "training_legitimate": int((in_training_split & (_df["Class"] == 0)).sum()),
        "training_fraud": int((in_training_split & (_df["Class"] == 1)).sum()),
    }


# ============================================================
# LOAD MODEL (no silent defaults)
# ============================================================

components = None
model_error = None

try:
    components = get_model_components()
except Exception as error:  # shown to the user below; detection is disabled
    model_error = str(error)

if model_error:
    st.error(
        "❌ **The saved model could not be loaded, so anomaly detection is disabled.**\n\n"
        f"{model_error}\n\n"
        "Check that `models/autoencoder.keras`, `models/scaler.pkl` and `models/threshold.txt` "
        "exist, then restart the dashboard."
    )

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Primary Model:** Autoencoder")

    if components is not None:
        model, scaler, threshold = components
        st.write("**Architecture:**")
        st.code(architecture_text(model))
        st.write("**Anomaly Threshold:**")
        st.write(f"{threshold:.4f}")
        st.caption("95th percentile of reconstruction errors on the legitimate training transactions.")
        st.success("Model loaded")
    else:
        st.error("Model not loaded")

    st.write("**Training:**")
    st.write("Legitimate transactions only (fraud labels not used)")

    st.write("**Detection Method:**")
    st.write("Reconstruction error above the threshold")

    st.divider()

    st.header("📖 Glossary")

    st.write(
        "**Anomaly:** a transaction the Autoencoder finds unusual "
        "(reconstruction error above the threshold). It needs review - "
        "it is **not** confirmed fraud."
    )

    st.write(
        "**Reconstruction error:** how badly the Autoencoder can rebuild a "
        "transaction. Unusual transactions are rebuilt badly."
    )

# ============================================================
# DATASET
# ============================================================

section_title("📁 Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file (optional - otherwise data/creditcard.csv is used)",
    type=["csv"]
)

df = None
dataset_id = None

if uploaded_file is not None:

    dataset_id = dataset_identity(uploaded_file, DEFAULT_DATA_PATH)
    dataset_label = f"uploaded file '{uploaded_file.name}'"

    try:
        df = read_uploaded_dataset(dataset_id, uploaded_file)
    except Exception as error:
        st.error(f"❌ The uploaded file could not be read as a CSV: {error}")

elif DEFAULT_DATA_PATH.exists():

    dataset_id = dataset_identity(None, DEFAULT_DATA_PATH)
    dataset_label = "default dataset data/creditcard.csv"
    df = read_default_dataset(str(DEFAULT_DATA_PATH), dataset_id)

else:

    st.info(
        "No dataset loaded. Upload a CSV file, or place the Kaggle dataset at "
        "`data/creditcard.csv`."
    )

is_default_dataset = uploaded_file is None

if df is not None:

    st.success(f"Loaded the {dataset_label}.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Transactions", f"{len(df):,}")

    with col2:
        if components is not None:
            required = list(components[1].feature_names_in_)
            present = sum(column in df.columns for column in required)
            st.metric("Model Input Columns Present", f"{present} of {len(required)}")
        else:
            st.metric("Columns", len(df.columns))

    with col3:
        st.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")

    with col4:
        if "Class" in df.columns and pd.api.types.is_numeric_dtype(df["Class"]):
            st.metric("Labelled Fraud (Class = 1)", f"{int(df['Class'].sum()):,}")
        else:
            st.metric("Labelled Fraud (Class = 1)", "N/A")

    st.caption(
        "The Class column (if present) is never used by the model. "
        "It is shown here only as information about the file."
    )

    st.markdown("**Preview (first 10 rows)**")

    st.dataframe(df.head(10), width="stretch")

# ============================================================
# ANOMALY DETECTION (whole loaded dataset)
# ============================================================

if df is not None:

    section_title("🔎 Anomaly Detection (whole loaded dataset)")

    threshold_text = f"{components[2]:.4f}" if components is not None else "unavailable"

    st.info(
        f"This scores **every row** in the loaded file with the saved Autoencoder. "
        f"An **anomaly** is a transaction whose reconstruction error is above the "
        f"threshold ({threshold_text}): it looks unusual, which is **not** the same as "
        f"confirmed fraud. These counts are **not** a measure of model accuracy."
    )

    if is_default_dataset and TEST_INDEX_PATH.exists() and "Class" in df.columns:
        split = training_split_counts(str(TEST_INDEX_PATH), dataset_id, df)
        st.warning(
            f"⚠️ This is the dataset the model was built from. "
            f"{split['training_split_rows']:,} of these rows (about 80%) belong to the training split. "
            f"Of those, only the **{split['training_legitimate']:,} legitimate transactions were used to "
            f"train the Autoencoder and set its threshold**; the {split['training_fraud']:,} fraudulent "
            f"training-split transactions were excluded from training. Because the threshold is the "
            f"95th percentile of those {split['training_legitimate']:,} transactions' errors, it flags "
            f"about 5% of them by design. For real performance on unseen data, see the "
            f"**Baseline Evaluation** tab ({split['test_rows']:,} held-out test transactions)."
        )

    st.caption(
        "Note: the `Time` column is 'seconds since the first transaction of that file', so "
        "values in a different file are not directly comparable with the training data."
    )

    detect_clicked = st.button(
        "🔍 Detect Anomalies",
        type="primary",
        width="stretch",
        disabled=components is None,
        key="detect_button"
    )

    if detect_clicked:

        # Any earlier result is removed first, so a failed run never leaves
        # old results on screen.
        st.session_state.pop("detection", None)

        with st.spinner("Scoring transactions..."):

            try:

                _, ignored_columns = select_model_features(df, components[1])

                results = detect_anomalies(df, components)

                st.session_state["detection"] = {
                    "dataset_id": dataset_id,
                    "results": results,
                    "ignored_columns": ignored_columns,
                }

                st.success("Anomaly detection completed.")

            except InputDataError as error:

                st.error(f"❌ This file cannot be scored: {error}")

            except Exception as error:

                st.error(f"❌ Unexpected error during anomaly detection: {error}")

# ------------------------------------------------------------
# Only show results that belong to the dataset loaded right now
# ------------------------------------------------------------

detection = st.session_state.get("detection")

if detection is not None and detection["dataset_id"] != dataset_id:

    st.session_state.pop("detection", None)
    detection = None

    if df is not None:
        st.info(
            "A different dataset is loaded now, so the previous results were cleared. "
            "Click **Detect Anomalies** to analyse this dataset."
        )

results = detection["results"] if detection is not None else None

# ============================================================
# DISPLAY DETECTION RESULTS
# ============================================================

if results is not None:

    threshold = components[2]

    section_title("📊 Detection Results (whole loaded dataset)")

    if detection["ignored_columns"]:
        st.info(
            "These extra columns were ignored by the model: "
            + ", ".join(f"`{c}`" for c in detection["ignored_columns"])
            + ". Old result columns are recalculated."
        )

    total_transactions = len(results)
    anomaly_count = int(results["anomaly"].sum())
    normal_count = total_transactions - anomaly_count
    anomaly_rate = anomaly_count / total_transactions * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Transactions Scored", f"{total_transactions:,}")

    with col2:
        st.metric("Not Flagged", f"{normal_count:,}")

    with col3:
        st.metric("🚨 Anomalies Detected", f"{anomaly_count:,}")

    with col4:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Flagged vs Not Flagged")

        chart_data = pd.DataFrame({
            "Result": ["Not flagged", "Anomaly (flagged)"],
            "Count": [normal_count, anomaly_count],
        })

        flagged_chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("Result:N", title="Result", sort=None),
                y=alt.Y("Count:Q", title="Number of transactions"),
                color=alt.Color(
                    "Result:N",
                    scale=alt.Scale(
                        domain=["Not flagged", "Anomaly (flagged)"],
                        range=[NORMAL_COLOR, ANOMALY_COLOR],
                    ),
                    legend=None,
                ),
                tooltip=["Result", alt.Tooltip("Count:Q", format=",")],
            )
            .properties(height=400)
        )

        st.altair_chart(flagged_chart, width="stretch")

    with col2:

        st.subheader("Reconstruction Error Distribution")

        histogram = log_histogram(results["reconstruction_error"], threshold)

        bars = (
            alt.Chart(histogram)
            .mark_bar()
            .encode(
                x=alt.X(
                    "bin_start:Q",
                    scale=alt.Scale(type="log"),
                    title="Reconstruction error (log scale)",
                ),
                x2="bin_end:Q",
                y=alt.Y(
                    "count:Q",
                    stack="zero",
                    scale=alt.Scale(type="symlog"),
                    title="Number of transactions (compressed scale)",
                ),
                color=alt.Color(
                    "group:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Not flagged", "Anomaly (flagged)"],
                        range=[NORMAL_COLOR, ANOMALY_COLOR],
                    ),
                    legend=alt.Legend(orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("bin_start:Q", title="From", format=".4f"),
                    alt.Tooltip("bin_end:Q", title="To", format=".4f"),
                    alt.Tooltip("group:N", title="Result"),
                    alt.Tooltip("count:Q", title="Transactions", format=","),
                ],
            )
        )

        threshold_line = (
            alt.Chart(pd.DataFrame({"threshold": [threshold]}))
            .mark_rule(color="black", strokeDash=[6, 4], size=2)
            .encode(x="threshold:Q", tooltip=[alt.Tooltip("threshold:Q", title="Threshold", format=".4f")])
        )

        threshold_label = (
            alt.Chart(pd.DataFrame({"threshold": [threshold], "label": [f"Threshold {threshold:.4f}"]}))
            .mark_text(align="left", dx=5, dy=-170, fontSize=12)
            .encode(x="threshold:Q", text="label:N")
        )

        st.altair_chart(
            (bars + threshold_line + threshold_label).properties(height=400),
            width="stretch"
        )

        st.caption(
            f"Every one of the {total_transactions:,} transactions is included - nothing is cut off. "
            "Bars to the right of the dashed line are anomalies. The vertical axis is compressed "
            "so that small bars stay visible."
        )

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 All Scored Transactions",
        "🚨 Anomalies Detected",
        "📊 Baseline Evaluation (held-out test set)",
        "🤖 Model Comparison (stored results)"
    ]
)

# ============================================================
# TAB 1 — ALL SCORED TRANSACTIONS
# ============================================================

with tab1:

    st.subheader("All Scored Transactions")

    if results is None:

        st.info("Run **Detect Anomalies** to see the scored transactions.")

    else:

        display_columns = [
            column
            for column in ["Time", "Amount", "reconstruction_error", "anomaly"]
            if column in results.columns
        ]

        st.dataframe(results[display_columns], width="stretch")

# ============================================================
# TAB 2 — ANOMALIES DETECTED
# ============================================================

with tab2:

    st.subheader("🚨 Anomalies Detected")

    if results is None:

        st.info("Run **Detect Anomalies** to see the flagged transactions.")

    else:

        anomalies = results[results["anomaly"] == 1].sort_values(
            "reconstruction_error", ascending=False
        )

        if len(anomalies) > 0:

            st.write(
                f"**{len(anomalies):,} transactions were flagged as anomalies** "
                "(sorted from highest to lowest reconstruction error)."
            )

            st.caption(
                "Each of these needs human review. An anomaly is unusual compared with the "
                "legitimate training transactions - it is not proof of fraud."
            )

            table = anomalies.copy()
            table["flag_reason"] = (
                "Reconstruction error above threshold (" + f"{threshold:.4f}" + ")"
            )

            display_columns = [
                column
                for column in ["Time", "Amount", "reconstruction_error", "anomaly", "flag_reason"]
                if column in table.columns
            ]

            st.dataframe(table[display_columns], width="stretch")

            st.download_button(
                label="⬇️ Download anomalies (CSV)",
                data=build_download(anomalies, threshold),
                file_name="intellifraud_anomalies.csv",
                mime="text/csv"
            )

            st.caption(
                "The downloaded file can be uploaded again: the model only reads its own "
                "30 input columns."
            )

        else:

            st.info("No anomalies were detected.")

# ============================================================
# TAB 3 — BASELINE EVALUATION (stored, verified)
# ============================================================

with tab3:

    st.subheader("📊 Baseline Evaluation of the Original Autoencoder")

    if not BASELINE_METRICS_PATH.exists():

        st.warning(
            "Verified baseline results not found (`results/metrics/baseline_metrics.csv`). "
            "Run `python src/baseline_evaluation.py` to create them."
        )

    else:

        metrics = read_baseline_metrics(str(BASELINE_METRICS_PATH))

        def value(name):
            return metrics[name]["value"]

        def ci(name, percent=True):
            low, high = metrics[name]["low"], metrics[name]["high"]
            if pd.isna(low):
                return None
            return f"95% CI {low:.2%} - {high:.2%}" if percent else f"95% CI {low:.4f} - {high:.4f}"

        st.info(
            f"These are the **stored results of the original, verified baseline evaluation** on "
            f"**{int(value('test_rows')):,} held-out test transactions** "
            f"({int(value('actual_normal')):,} legitimate, {int(value('actual_fraud')):,} fraud) "
            f"that the model never trained on. They do **not** change when you upload a "
            f"different dataset or click Detect Anomalies."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Recall (fraud caught)", f"{value('recall'):.2%}")
            st.caption(ci("recall"))

        with col2:
            st.metric("Precision (alerts that are fraud)", f"{value('precision'):.2%}")
            st.caption(ci("precision"))

        with col3:
            st.metric("F1 Score", f"{value('f1_score'):.2%}")
            st.caption(ci("f1_score"))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Average Precision", f"{value('average_precision'):.4f}")
            st.caption(ci("average_precision", percent=False))

        with col2:
            st.metric("PR-AUC (trapezoidal)", f"{value('pr_auc_trapezoidal'):.4f}")
            st.caption(ci("pr_auc_trapezoidal", percent=False))

        with col3:
            st.metric("Random-guess PR-AUC", f"{value('no_skill_pr_auc'):.4f}")
            st.caption("Equal to the fraud rate in the test set")

        st.caption(
            f"Accuracy: {value('accuracy'):.2%} - lower than the {value('always_normal_accuracy'):.2%} a "
            f"model would get by never flagging anything, so accuracy is not a useful measure here."
        )

        # ------------------------------------------------
        # CONFUSION MATRIX
        # ------------------------------------------------

        st.markdown("### 🔲 Confusion Matrix (held-out test set)")

        tn = int(value("true_negatives"))
        fp = int(value("false_positives"))
        fn = int(value("false_negatives"))
        tp = int(value("true_positives"))

        confusion_matrix_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual legitimate", "Actual fraud"],
            columns=["Predicted normal", "Predicted anomaly"]
        )

        st.dataframe(confusion_matrix_df, width="stretch")

        # ------------------------------------------------
        # INTERPRETATION (calculated from the numbers)
        # ------------------------------------------------

        st.markdown("### 💡 Interpretation")

        st.write(
            f"- On the held-out test set the Autoencoder caught **{tp} of {tp + fn} fraud cases** "
            f"(recall {value('recall'):.2%}) and missed {fn}.\n"
            f"- It also flagged **{fp:,} legitimate transactions** - about "
            f"**{value('false_alarms_per_fraud_caught'):.1f} false alarms for every fraud caught** "
            f"(precision {value('precision'):.2%}).\n"
            f"- Average Precision is {value('average_precision'):.4f} compared with "
            f"{value('no_skill_pr_auc'):.4f} for random guessing.\n"
            f"- The Autoencoder was trained without fraud labels; labels were used only "
            f"afterwards to measure these results."
        )

        st.caption(
            f"Threshold {value('threshold'):.4f}, unchanged. Full method, duplicate sensitivity "
            f"analysis and limitations: `{BASELINE_REPORT_PATH}`."
        )

# ============================================================
# TAB 4 — MODEL COMPARISON (stored)
# ============================================================

with tab4:

    st.subheader("🤖 Unsupervised Model Comparison")

    if not COMPARISON_PATH.exists():

        st.warning(
            "Model comparison results not found (`models/model_comparison.csv`). "
            "They are created by `src/compare_models.py`."
        )

    else:

        comparison_df = read_comparison(str(COMPARISON_PATH))

        st.info(
            "Stored results from `src/compare_models.py`, measured on the same held-out test "
            "set. They are **not** recalculated by the dashboard."
        )

        # ------------------------------------------------
        # COMPARISON TABLE
        # ------------------------------------------------

        st.markdown("### 📋 Performance Comparison")

        formatted_comparison = comparison_df.copy()

        for column in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            formatted_comparison[column] = formatted_comparison[column].map(lambda x: f"{x:.2%}")

        st.dataframe(formatted_comparison, width="stretch", hide_index=True)

        # ------------------------------------------------
        # MODEL PERFORMANCE CHART
        # ------------------------------------------------

        st.markdown("### 📈 Model Performance")

        performance_long = comparison_df[["Model", "Precision", "Recall", "F1 Score"]].melt(
            id_vars="Model",
            value_vars=["Precision", "Recall", "F1 Score"],
            var_name="Metric",
            value_name="Score"
        )

        performance_chart = (
            alt.Chart(performance_long)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", title="Model"),
                xOffset="Metric:N",
                y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Metric:N", title="Metric"),
                tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".2%")]
            )
            .properties(height=400)
        )

        st.altair_chart(performance_chart, width="stretch")

        # ------------------------------------------------
        # FRAUD CAUGHT CHART
        # ------------------------------------------------

        st.markdown("### 🎯 Known Fraud Cases Caught (held-out test set)")

        fraud_caught_chart = (
            alt.Chart(comparison_df[["Model", "Fraud Caught"]])
            .mark_bar()
            .encode(
                x=alt.X("Model:N", title="Model"),
                y=alt.Y("Fraud Caught:Q", title="Known fraud cases caught"),
                color=alt.Color("Model:N", title="Model", legend=None),
                tooltip=["Model", "Fraud Caught"]
            )
            .properties(height=400)
        )

        st.altair_chart(fraud_caught_chart, width="stretch")

        # ------------------------------------------------
        # HONEST SUMMARY
        # ------------------------------------------------

        st.markdown("### 🏆 Is One Model Better?")

        f1_ci = None
        if BASELINE_METRICS_PATH.exists():
            f1 = read_baseline_metrics(str(BASELINE_METRICS_PATH))["f1_score"]
            f1_ci = (f1["low"], f1["high"])

        st.write(comparison_summary(comparison_df, f1_ci))

        st.caption(
            "Note: One-Class SVM was trained on a random sample of only 3,000 legitimate "
            "training transactions (for speed), while the Autoencoder and Isolation Forest "
            "used all of them, so the comparison is not fully equal."
        )

        # ------------------------------------------------
        # MODEL EXPLANATION
        # ------------------------------------------------

        st.markdown("### 💡 Why Compare These Models?")

        st.write(
            """
            **Autoencoder:** Learns to reconstruct normal
            transaction patterns. Transactions with high
            reconstruction error are considered anomalies.

            **Isolation Forest:** Identifies observations that
            are easier to isolate from the rest of the data.

            **One-Class SVM:** Learns the boundary of normal
            transactions and identifies observations outside
            that boundary.

            All three approaches are unsupervised and do not
            require fraud labels during model training.
            """
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🔍 IntelliFraud |
        Unsupervised Anomaly Detection with an Autoencoder |
        Flagged transactions are anomalies for review, not confirmed fraud
    </div>
    """,
    unsafe_allow_html=True
)
