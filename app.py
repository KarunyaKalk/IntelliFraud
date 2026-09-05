import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

from src.anomaly_detection import (
    detect_anomalies,
    load_model_components
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
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
    '<div class="main-title">🔍 Credit Card Fraud Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Unsupervised Fraud Detection using Autoencoder, '
    'Isolation Forest and One-Class SVM'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL INFORMATION
# ============================================================

try:

    model, scaler, threshold = load_model_components()

    model_loaded = True

except Exception:

    model_loaded = False
    threshold = 0

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Primary Model:** Autoencoder")

    st.write("**Architecture:**")
    st.code("30 → 16 → 8 → 16 → 30")

    st.write("**Training:**")
    st.write("Normal transactions only")

    st.write("**Detection Method:**")
    st.write("Reconstruction Error")

    if model_loaded:

        st.write("**Anomaly Threshold:**")
        st.write(f"{threshold:.4f}")

    st.divider()

    st.header("📊 Dataset Features")

    st.write("30 transaction features")

    st.write(
        "Class labels are used only for evaluation, "
        "not for model training."
    )

# ============================================================
# UPLOAD DATASET
# ============================================================

st.markdown(
    '<div class="section-title">📁 Upload Transaction Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

else:

    default_path = Path("data/creditcard.csv")

    if default_path.exists():

        df = pd.read_csv(default_path)

    else:

        df = None

# ============================================================
# DATASET SECTION
# ============================================================

if df is not None:

    st.success("Dataset loaded successfully.")

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📋 Dataset Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Transactions",
            f"{len(df):,}"
        )

    with col2:

        feature_count = len(df.columns)

        if "Class" in df.columns:
            feature_count -= 1

        st.metric(
            "Features",
            feature_count
        )

    with col3:

        st.metric(
            "Missing Values",
            f"{df.isnull().sum().sum():,}"
        )

    with col4:

        if "Class" in df.columns:

            fraud_count = int(df["Class"].sum())

            st.metric(
                "Known Fraud Cases",
                f"{fraud_count:,}"
            )

        else:

            st.metric(
                "Known Fraud Cases",
                "N/A"
            )

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">👀 Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # ========================================================
    # DETECTION
    # ========================================================

    st.markdown(
        '<div class="section-title">🚨 Fraud Detection</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "🔍 Detect Fraud",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing transactions..."
        ):

            try:

                results = detect_anomalies(df)

                st.session_state["results"] = results

                st.success(
                    "Fraud detection completed successfully!"
                )

            except Exception as e:

                st.error(
                    f"Error during fraud detection: {e}"
                )

# ============================================================
# DISPLAY DETECTION RESULTS
# ============================================================

if "results" in st.session_state:

    results = st.session_state["results"]

    st.markdown(
        '<div class="section-title">📊 Detection Results</div>',
        unsafe_allow_html=True
    )

    total_transactions = len(results)

    anomaly_count = int(
        results["anomaly"].sum()
    )

    normal_count = (
        total_transactions - anomaly_count
    )

    # --------------------------------------------------------
    # ANOMALY RATE
    # --------------------------------------------------------

    anomaly_rate = (
        anomaly_count / total_transactions * 100
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

    with col2:

        st.metric(
            "Normal Transactions",
            f"{normal_count:,}"
        )

    with col3:

        st.metric(
            "🚨 Anomalies Detected",
            f"{anomaly_count:,}"
        )

    with col4:

        st.metric(
            "Anomaly Rate",
            f"{anomaly_rate:.2f}%"
        )

    # ========================================================
    # DETECTION VISUALIZATION
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Detection Visualization</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # NORMAL VS ANOMALY CHART
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "Normal vs Anomalous Transactions"
        )

        chart_data = pd.DataFrame(
            {
                "Transaction Type": [
                    "Normal",
                    "Anomaly"
                ],
                "Count": [
                    normal_count,
                    anomaly_count
                ]
            }
        )

        normal_anomaly_chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Transaction Type:N",
                    title="Transaction Type"
                ),
                y=alt.Y(
                    "Count:Q",
                    title="Number of Transactions"
                ),
                tooltip=[
                    "Transaction Type",
                    "Count"
                ]
            )
            .properties(
                height=400
            )
        )

        st.altair_chart(
            normal_anomaly_chart,
            use_container_width=True
        )

    # --------------------------------------------------------
    # RECONSTRUCTION ERROR CHART
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "Reconstruction Error Distribution"
        )

        error_data = results[
            "reconstruction_error"
        ]

        upper_limit = error_data.quantile(
            0.99
        )

        clipped_errors = error_data[
            error_data <= upper_limit
        ]

        histogram_data = pd.DataFrame(
            {
                "Reconstruction Error":
                    clipped_errors
            }
        )

        reconstruction_chart = (
            alt.Chart(histogram_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Reconstruction Error:Q",
                    bin=alt.Bin(maxbins=50),
                    title="Reconstruction Error"
                ),
                y=alt.Y(
                    "count():Q",
                    title="Number of Transactions"
                ),
                tooltip=[
                    alt.Tooltip(
                        "count():Q",
                        title="Transactions"
                    )
                ]
            )
            .properties(
                height=400
            )
        )

        st.altair_chart(
            reconstruction_chart,
            use_container_width=True
        )

    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📋 All Transactions",
            "🚨 Fraud Transactions",
            "📊 Evaluation Metrics",
            "🤖 Model Comparison"
        ]
    )

    # ========================================================
    # TAB 1 — ALL TRANSACTIONS
    # ========================================================

    with tab1:

        st.subheader(
            "All Transaction Results"
        )

        display_columns = [
            column
            for column in [
                "Time",
                "Amount",
                "reconstruction_error",
                "anomaly"
            ]
            if column in results.columns
        ]

        st.dataframe(
            results[display_columns],
            use_container_width=True
        )

    # ========================================================
    # TAB 2 — FRAUD TRANSACTIONS
    # ========================================================

    with tab2:

        st.subheader(
            "🚨 Flagged Transactions"
        )

        anomalies = results[
            results["anomaly"] == 1
        ].copy()


        if len(anomalies) > 0:

            anomalies["Why Flagged"] = (
                "Reconstruction error exceeds "
                f"threshold ({threshold:.4f})"
            )

            display_columns = [
                column
                for column in [
                    "Time",
                    "Amount",
                    "reconstruction_error",
                    "anomaly",
                    "Why Flagged"
                ]
                if column in anomalies.columns
            ]

            st.write(
                f"**{len(anomalies):,} transactions "
                "were flagged as anomalies.**"
            )

            st.dataframe(
                anomalies[display_columns],
                use_container_width=True
            )

            csv = anomalies.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Flagged Transactions",
                data=csv,
                file_name="flagged_transactions.csv",
                mime="text/csv"
            )

        else:

            st.info(
                "No anomalies were detected."
            )

    # ========================================================
    # TAB 3 — EVALUATION METRICS
    # ========================================================

    with tab3:

        st.subheader(
            "📊 Autoencoder Evaluation"
        )

        evaluation_path = Path(
            "models/evaluation_results.csv"
        )


        if evaluation_path.exists():

            evaluation_df = pd.read_csv(
                evaluation_path
            )

            metrics = dict(
                zip(
                    evaluation_df["Metric"],
                    evaluation_df["Value"]
                )
            )

            # ------------------------------------------------
            # MAIN METRIC CARDS
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Accuracy",
                    f"{float(metrics['Accuracy']) * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Precision",
                    f"{float(metrics['Precision']) * 100:.2f}%"
                )

            with col3:

                st.metric(
                    "Recall",
                    f"{float(metrics['Recall']) * 100:.2f}%"
                )

            with col4:

                st.metric(
                    "F1 Score",
                    f"{float(metrics['F1 Score']) * 100:.2f}%"
                )

            # ------------------------------------------------
            # FRAUD DETECTION SUMMARY
            # ------------------------------------------------

            st.markdown(
                "### 🎯 Fraud Detection Summary"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Actual Fraud Cases",
                    int(metrics["Actual Fraud Cases"])
                )

            with col2:

                st.metric(
                    "Fraud Cases Caught",
                    int(metrics["True Positives"])
                )

            with col3:

                st.metric(
                    "Missed Fraud Cases",
                    int(metrics["False Negatives"])
                )

            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            st.markdown(
                "### 🔲 Confusion Matrix"
            )

            tn = int(metrics["True Negatives"])
            fp = int(metrics["False Positives"])
            fn = int(metrics["False Negatives"])
            tp = int(metrics["True Positives"])


            confusion_matrix_df = pd.DataFrame(
                [
                    [tn, fp],
                    [fn, tp]
                ],
                index=[
                    "Actual Normal",
                    "Actual Fraud"
                ],
                columns=[
                    "Predicted Normal",
                    "Predicted Fraud"
                ]
            )

            st.dataframe(
                confusion_matrix_df,
                use_container_width=True
            )

            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            st.markdown(
                "### 💡 Interpretation"
            )

            st.info(
                f"""
                The Autoencoder detected **{tp} out of
                {int(metrics["Actual Fraud Cases"])} known fraud cases**.

                **Recall:** {float(metrics["Recall"]) * 100:.2f}%

                This means the model successfully identified most
                of the known fraud cases.

                **Precision:** {float(metrics["Precision"]) * 100:.2f}%

                The lower precision is mainly due to false alarms,
                which is expected when detecting very rare fraud cases.

                The Autoencoder was trained without using the fraud
                labels. The labels were used only after prediction
                to evaluate the model.
                """
            )

        else:

            st.warning(
                "Evaluation results file not found. "
                "Run evaluate.py first."
            )

    # ========================================================
    # TAB 4 — MODEL COMPARISON
    # ========================================================

    with tab4:

        st.subheader(
            "🤖 Unsupervised Model Comparison"
        )

        comparison_path = Path(
            "models/model_comparison.csv"
        )

        if comparison_path.exists():

            comparison_df = pd.read_csv(
                comparison_path
            )

            # ------------------------------------------------
            # COMPARISON TABLE
            # ------------------------------------------------

            st.markdown(
                "### 📋 Performance Comparison"
            )

            formatted_comparison = comparison_df.copy()

            formatted_comparison[
                "Accuracy"
            ] = formatted_comparison[
                "Accuracy"
            ].map(
                lambda x: f"{x:.2%}"
            )

            formatted_comparison[
                "Precision"
            ] = formatted_comparison[
                "Precision"
            ].map(
                lambda x: f"{x:.2%}"
            )

            formatted_comparison[
                "Recall"
            ] = formatted_comparison[
                "Recall"
            ].map(
                lambda x: f"{x:.2%}"
            )

            formatted_comparison[
                "F1 Score"
            ] = formatted_comparison[
                "F1 Score"
            ].map(
                lambda x: f"{x:.2%}"
            )

            st.dataframe(
                formatted_comparison,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # MODEL PERFORMANCE CHART
            # ------------------------------------------------

            st.markdown(
                "### 📈 Model Performance"
            )

            performance_df = comparison_df[
                [
                    "Model",
                    "Precision",
                    "Recall",
                    "F1 Score"
                ]
            ]

            performance_long = performance_df.melt(
                id_vars="Model",
                value_vars=[
                    "Precision",
                    "Recall",
                    "F1 Score"
                ],
                var_name="Metric",
                value_name="Score"
            )

            performance_chart = (
                alt.Chart(performance_long)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Model:N",
                        title="Model"
                    ),
                    y=alt.Y(
                        "Score:Q",
                        title="Score",
                        scale=alt.Scale(
                            domain=[0, 1]
                        )
                    ),
                    color=alt.Color(
                        "Metric:N",
                        title="Metric"
                    ),
                    tooltip=[
                        "Model",
                        "Metric",
                        alt.Tooltip(
                            "Score:Q",
                            format=".2%"
                        )
                    ]
                )
                .properties(
                    height=400
                )
            )

            st.altair_chart(
                performance_chart,
                use_container_width=True
            )

            # ------------------------------------------------
            # FRAUD CAUGHT CHART
            # ------------------------------------------------

            st.markdown(
                "### 🎯 Known Fraud Cases Caught"
            )

            fraud_chart = comparison_df[
                [
                    "Model",
                    "Fraud Caught"
                ]
            ]

            fraud_caught_chart = (
                alt.Chart(fraud_chart)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Model:N",
                        title="Model"
                    ),
                    y=alt.Y(
                        "Fraud Caught:Q",
                        title="Known Fraud Cases Caught"
                    ),
                    color=alt.Color(
                        "Model:N",
                        title="Model",
                        legend=None
                    ),
                    tooltip=[
                        "Model",
                        "Fraud Caught"
                    ]
                )
                .properties(
                    height=400
                )
            )

            st.altair_chart(
                fraud_caught_chart,
                use_container_width=True
            )

            # ------------------------------------------------
            # BEST MODEL
            # ------------------------------------------------

            best_model_row = comparison_df.loc[
                comparison_df["F1 Score"].idxmax()
            ]

            st.markdown(
                "### 🏆 Model Selection"
            )

            st.success(
                f"Based on F1 Score, the best performing model "
                f"in this comparison is **{best_model_row['Model']}**."
            )

            # ------------------------------------------------
            # MODEL EXPLANATION
            # ------------------------------------------------

            st.markdown(
                "### 💡 Why Compare These Models?"
            )

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

        else:

            st.warning(
                "Model comparison results not found. "
                "Run compare_models.py first."
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🔍 Credit Card Fraud Detection |
        Unsupervised Machine Learning |
        Autoencoder + Isolation Forest + One-Class SVM
    </div>
    """,
    unsafe_allow_html=True
)