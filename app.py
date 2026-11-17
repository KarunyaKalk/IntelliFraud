"""
THE FRAUD BUREAU - IntelliFraud dashboard (Streamlit).

Run from the project folder:
    source .venv/bin/activate
    streamlit run app.py

Layout
------
Sidebar : Workspace - choose the dataset (original / synthetic 5%) and the
          model (baseline / improved v2); shows model status and a glossary.
Tabs    : 1. Dataset Intelligence        - live overview of the selected or uploaded data
          2. Transaction Investigations  - score every row, live charts, case files, downloads
          3. Bureau Intelligence Report  - stored, verified evaluation of both models

Key rules
---------
* A flagged transaction is an ANOMALY (unusual), not confirmed fraud.
* The Class column is never a model input; it is only compared afterwards.
* Each model always uses its own model + scaler + threshold files.
* Official results are read from results/ and never recalculated. Report
  charts are drawn live from the stored test scores, metric files and
  training histories; they only reshape stored values for plotting.
* The synthetic dataset contains artificial fraud rows: demonstration use only.

Caching note: models are loaded once per dashboard session
(st.cache_resource). If files in models/ change, restart the dashboard or
use the menu's "Clear cache". Colours and fonts live in .streamlit/config.toml.
"""

import html
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.anomaly_detection import (
    InputDataError,
    detect_anomalies,
    load_model_components,
    select_model_features,
)
from src.dashboard_utils import (
    DATASET_OPTIONS,
    MODEL_OPTIONS,
    amount_histogram,
    amount_statistics,
    anomaly_score_bands,
    architecture_text,
    build_case_file,
    build_download,
    case_context,
    ci_frame,
    class_distribution,
    class_error_histogram,
    compact_results_csv,
    comparison_summary,
    confusion_frame,
    dataset_identity,
    dataset_summary,
    flagged_positions,
    format_metric,
    improved_training_counts,
    label_comparison,
    load_official_comparison,
    log_histogram,
    model_file_paths,
    paired_difference_frame,
    pr_curve_points,
    read_synthetic_metadata,
    shared_log_edges,
    synthetic_start_row,
    training_history,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_INDEX_PATH = PROJECT_ROOT / "models" / "test_indices.csv"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
BASELINE_METRICS_PATH = METRICS_DIR / "baseline_metrics.csv"
IMPROVED_METRICS_PATH = METRICS_DIR / "improved_metrics.csv"
PAIRED_COMPARISON_PATH = METRICS_DIR / "model_comparison_baseline_vs_improved.csv"
BASELINE_SCORES_PATH = METRICS_DIR / "baseline_test_scores.csv"
IMPROVED_SCORES_PATH = METRICS_DIR / "improved_test_scores.csv"
BASELINE_HISTORY_PATH = PROJECT_ROOT / "models" / "training_history.csv"
IMPROVED_HISTORY_PATH = PROJECT_ROOT / "models" / "improved" / "training_history.csv"
THREE_MODEL_COMPARISON_PATH = PROJECT_ROOT / "models" / "model_comparison.csv"

# Palette (same values as .streamlit/config.toml). The five chart hues were
# checked for contrast and colour-vision separation on the dark panel colour.
CANVAS = "#0F1114"
PANEL = "#15181C"
PANEL_RAISED = "#1B1F25"
BORDER = "#262B32"
GRID = "#1F242A"
TEXT = "#E8EAED"
TEXT_2 = "#A6ADB7"
MUTED = "#6E7681"
GREEN = "#2DAE78"      # legitimate / not flagged
RED = "#D8343F"        # fraud / flagged anomaly
AMBER = "#B9851A"      # synthetic (generated) fraud rows
BLUE = "#4C86D9"       # baseline model
GOLD = "#B48C3C"       # Improved v2 model

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
MONO = "'SF Mono', 'IBM Plex Mono', Menlo, Consolas, monospace"

GROUP_COLOURS = {"Legitimate": GREEN, "Fraud": RED, "Fraud (original)": RED, "Fraud (synthetic)": AMBER,
                 "All transactions": GREEN}
RESULT_DOMAIN = ["Not flagged", "Anomaly (flagged)"]
RESULT_RANGE = [GREEN, RED]
MODEL_DOMAIN = ["Baseline", "Improved v2"]
MODEL_RANGE = [BLUE, GOLD]

CASE_FILE_CHOICES = 200   # how many flagged transactions can be opened as case files

# ============================================================
# PAGE CONFIGURATION AND STYLE
# ============================================================

st.set_page_config(
    page_title="The Fraud Bureau | IntelliFraud",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"], .block-container {
        padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1400px; }
    [data-testid="stHeader"] { background: transparent; }
    body, [data-testid="stAppViewContainer"] { font-feature-settings: "tnum" 1, "cv11" 1; }
    h1, h2, h3, h4 { letter-spacing: -0.01em; }

    /* ---------- Brand bar ---------- */
    .fb-top { display: flex; justify-content: space-between; align-items: center; gap: 18px; flex-wrap: wrap;
              padding: 2px 0 16px; border-bottom: 1px solid #262B32; margin-bottom: 4px; }
    .fb-brand { display: flex; align-items: center; gap: 14px; min-width: 0; }
    .fb-mark { width: 38px; height: 38px; border: 1px solid #323840; border-radius: 8px; background: #171A1F;
               display: flex; align-items: center; justify-content: center; flex: none; }
    .fb-mark span { width: 14px; height: 14px; border: 2px solid #2DAE78; border-radius: 3px;
                    box-shadow: 5px 5px 0 -2px #D8343F; }
    .fb-name { font-size: 15px; font-weight: 650; color: #E8EAED; letter-spacing: 0.08em; white-space: nowrap; }
    .fb-name .product { color: #A6ADB7; font-weight: 500; letter-spacing: 0.01em; }
    .fb-name .sep { color: #3A4048; margin: 0 8px; font-weight: 400; }
    .fb-sub { font-size: 12.5px; color: #8A929D; margin-top: 2px; }
    .fb-chips { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .fb-chip { font-size: 12px; color: #A6ADB7; border: 1px solid #262B32; background: #15181C; border-radius: 6px;
               padding: 5px 10px; white-space: nowrap; }
    .fb-chip b { color: #E8EAED; font-weight: 550; }
    .fb-chip .k { color: #6E7681; margin-right: 6px; }
    .fb-status { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; }
    .fb-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 7px;
              vertical-align: 1px; }
    .fb-dot.ok { background: #2DAE78; box-shadow: 0 0 0 3px rgba(45,174,120,0.16); }
    .fb-dot.warn { background: #B9851A; } .fb-dot.fail { background: #D8343F; }

    /* ---------- Tabs ---------- */
    [data-baseweb="tab-list"] { gap: 26px; border-bottom: 1px solid #262B32; }
    [data-baseweb="tab"] { padding: 10px 0 10px; font-size: 13.5px; background: transparent; }
    [data-baseweb="tab"] p { font-size: 13.5px; font-weight: 500; }
    [data-baseweb="tab"][aria-selected="false"] p { color: #8A929D; }
    [data-baseweb="tab-highlight"] { background-color: #E8EAED; height: 2px; }
    [data-baseweb="tab-border"] { display: none; }

    /* ---------- Section headings ---------- */
    .fb-h { margin: 18px 0 10px; }
    .fb-h .t { font-size: 17px; font-weight: 600; color: #E8EAED; }
    .fb-h .d { font-size: 12.5px; color: #8A929D; margin-top: 2px; max-width: 900px; line-height: 1.5; }

    /* ---------- Panels (keyed bordered containers) ---------- */
    div[class*="st-key-panel_"] { background: #15181C; border-color: #262B32 !important; border-radius: 8px; }
    .fb-card-h { display: flex; justify-content: space-between; align-items: baseline; gap: 10px;
                 margin-bottom: 2px; }
    .fb-card-h .t { font-size: 13.5px; font-weight: 600; color: #E8EAED; }
    .fb-card-h .s { font-size: 11.5px; color: #6E7681; text-align: right; }
    .fb-card-d { font-size: 12px; color: #8A929D; margin: 0 0 6px; line-height: 1.5; }

    /* ---------- KPI cards (st.metric) ---------- */
    [data-testid="stMetric"] { background: #15181C; border: 1px solid #262B32; border-radius: 8px;
                               padding: 12px 14px 10px; }
    [data-testid="stMetricLabel"] p { font-size: 12px; color: #8A929D; font-weight: 500; }
    [data-testid="stMetricValue"] { font-weight: 600; color: #E8EAED; letter-spacing: -0.01em; }

    /* ---------- Report comparison cards ---------- */
    .fb-kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin: 4px 0 14px; }
    @media (max-width: 1100px) { .fb-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    .fb-kpi { background: #15181C; border: 1px solid #262B32; border-radius: 8px; padding: 12px 14px; }
    .fb-kpi .l { font-size: 12px; color: #8A929D; }
    .fb-kpi .row { display: flex; justify-content: space-between; align-items: baseline; margin-top: 6px; }
    .fb-kpi .m { font-size: 11.5px; color: #8A929D; }
    .fb-kpi .m i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 6px; }
    .fb-kpi .v { font-size: 17px; font-weight: 600; color: #E8EAED; }
    .fb-kpi .diff { font-size: 11.5px; color: #8A929D; border-top: 1px solid #22272D; margin-top: 8px;
                    padding-top: 6px; }

    /* ---------- Notes and findings ---------- */
    .fb-note { background: #15181C; border: 1px solid #262B32; border-left: 2px solid #4A515B;
               border-radius: 6px; padding: 11px 14px; font-size: 13px; line-height: 1.6; color: #C5CAD2;
               margin-bottom: 10px; }
    .fb-note.green { border-left-color: #2DAE78; } .fb-note.red { border-left-color: #D8343F; }
    .fb-note.amber { border-left-color: #B9851A; }
    .fb-note .k { display: block; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; color: #8A929D;
                  text-transform: uppercase; margin-bottom: 3px; }
    .fb-note b { color: #E8EAED; font-weight: 600; }
    .fb-finding { font-size: 15px; font-weight: 600; color: #E8EAED; margin: 2px 0 4px; }

    /* ---------- Alerts: quieter, panel-like ---------- */
    [data-testid="stAlertContainer"] { border-radius: 6px; padding: 10px 14px; font-size: 13px;
        background: #15181C !important; border: 1px solid #262B32; border-left: 2px solid #4C86D9; }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { border-left-color: #B9851A; }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) { border-left-color: #D8343F; }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { border-left-color: #2DAE78; }
    [data-testid="stAlertContainer"] p, [data-testid="stAlertContainer"] li {
        font-size: 13px; line-height: 1.6; color: #C5CAD2 !important; }
    [data-testid="stAlertContainer"] strong { color: #E8EAED !important; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] { min-width: 272px; max-width: 290px; }
    [data-testid="stSidebarContent"] { padding-top: 6px; }
    .fb-side-k { font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
                 color: #6E7681; margin: 4px 0 6px; }
    .fb-side-brand { font-size: 13px; font-weight: 650; letter-spacing: 0.08em; color: #E8EAED; }
    .fb-side-brand span { color: #6E7681; font-weight: 500; letter-spacing: 0.02em; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size: 13px; }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: 12px; color: #8A929D; font-weight: 500; }
    .fb-status-list { border: 1px solid #22272D; border-radius: 8px; background: #111418; padding: 4px 12px; }
    .fb-status-list .row { display: flex; justify-content: space-between; gap: 10px; padding: 6px 0;
                           border-bottom: 1px solid #1C2026; font-size: 12px; }
    .fb-status-list .row:last-child { border-bottom: none; }
    .fb-status-list .k { color: #6E7681; } .fb-status-list .v { color: #D5D9DF; text-align: right; }
    .fb-status-list .ok { color: #2DAE78; } .fb-status-list .warn { color: #C99A3A; }
    .fb-status-list .fail { color: #E05A63; }

    /* ---------- Case file ---------- */
    .cf { background: #15181C; border: 1px solid #2C3138; border-radius: 10px; overflow: hidden; }
    .cf-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;
               padding: 16px 18px 14px; border-bottom: 1px solid #22272D; background: #171B20; }
    .cf-kicker { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: #6E7681; }
    .cf-ref { font-size: 21px; font-weight: 650; color: #E8EAED; margin-top: 2px; letter-spacing: 0.01em;
              font-family: """ + MONO + """; }
    .cf-meta { font-size: 12px; color: #8A929D; margin-top: 3px; }
    .cf-badge { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; padding: 5px 10px; border-radius: 5px;
                white-space: nowrap; }
    .cf-badge.flag { color: #F07A82; background: rgba(216,52,63,0.12); border: 1px solid rgba(216,52,63,0.45); }
    .cf-badge.normal { color: #5CC79A; background: rgba(45,174,120,0.10); border: 1px solid rgba(45,174,120,0.4); }
    .cf-figs { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border-bottom: 1px solid #22272D; }
    @media (max-width: 900px) { .cf-figs { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    .cf-fig { padding: 12px 18px; border-right: 1px solid #22272D; }
    .cf-fig:last-child { border-right: none; }
    .cf-fig .l { font-size: 11.5px; color: #8A929D; }
    .cf-fig .v { font-size: 18px; font-weight: 600; color: #E8EAED; margin-top: 3px; font-family: """ + MONO + """; }
    .cf-fig .v.hot { color: #F07A82; }
    .cf-gauge { padding: 12px 18px 14px; border-bottom: 1px solid #22272D; }
    .cf-gauge .bar { position: relative; height: 6px; background: #22272D; border-radius: 3px; margin: 8px 0 6px; }
    .cf-gauge .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; }
    .cf-gauge .tick { position: absolute; top: -5px; width: 2px; height: 16px; background: #E8EAED; }
    .cf-gauge .legend { display: flex; justify-content: space-between; font-size: 11.5px; color: #8A929D; }
    .cf table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .cf td { padding: 8px 18px; border-bottom: 1px solid #1E2228; vertical-align: top; }
    .cf td.k { width: 36%; color: #8A929D; }
    .cf td.v { color: #E0E3E8; }
    .cf-note { padding: 12px 18px 14px; font-size: 13px; color: #C5CAD2; line-height: 1.55; background: #13161A; }
    .cf-note .k { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: #6E7681;
                  margin-bottom: 3px; }

    /* ---------- Chart tooltips ---------- */
    #vg-tooltip-element { background: #1B1F25; color: #E8EAED; border: 1px solid #323840; border-radius: 6px;
                          font-family: Inter, -apple-system, sans-serif; font-size: 12px;
                          box-shadow: 0 8px 22px rgba(0,0,0,0.4); }
    #vg-tooltip-element table tr td.key { color: #A6ADB7; }

    .fb-footer { font-size: 11.5px; color: #6E7681; border-top: 1px solid #22272D; margin-top: 30px;
                 padding-top: 12px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SMALL LAYOUT HELPERS
# ============================================================

def display_path(path):
    """Show a path relative to the project folder when possible (never fails)."""
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def heading(title, description=""):
    extra = f'<div class="d">{description}</div>' if description else ""
    st.markdown(f'<div class="fb-h"><div class="t">{html.escape(title)}</div>{extra}</div>', unsafe_allow_html=True)


def note(content_html, accent=""):
    st.markdown(f'<div class="fb-note {accent}">{content_html}</div>', unsafe_allow_html=True)


def card_header(title, side="", description=""):
    st.markdown(
        f'<div class="fb-card-h"><span class="t">{html.escape(title)}</span>'
        f'<span class="s">{html.escape(side)}</span></div>'
        + (f'<div class="fb-card-d">{description}</div>' if description else ""),
        unsafe_allow_html=True,
    )


def panel(key):
    """A bordered charcoal card. The key gives it a CSS class (st-key-panel_...)."""
    return st.container(border=True, key=f"panel_{key}")


def status_list_html(rows):
    """Sidebar status block. rows: list of (key, value, state) with state ok / warn / fail / ''."""
    lines = [f'<div class="row"><span class="k">{html.escape(k)}</span>'
             f'<span class="v {state}">{html.escape(v)}</span></div>' for k, v, state in rows]
    return '<div class="fb-status-list">' + "".join(lines) + "</div>"


def case_file_markup(case, dataset_text):
    """A transaction case file as a document-like panel. Every value comes from `case`."""
    flagged = case["classification"] == "Anomaly"
    amount = "Not available" if case["amount"] is None else f"{case['amount']:,.2f}"
    ratio = case["error_to_threshold_ratio"]

    # Gauge: linear scale from 0 to a little past the larger of the error and the threshold.
    scale_end = max(ratio, 1.0) * 1.15
    fill_width = min(ratio / scale_end, 1.0) * 100
    tick_left = 1.0 / scale_end * 100
    fill_colour = RED if flagged else GREEN

    figures = [
        ("Transaction amount", amount, False),
        ("Reconstruction error", f"{case['reconstruction_error']:.6f}", flagged),
        ("Model threshold", f"{case['threshold']:.6f}", False),
        ("Error-to-threshold ratio", f"{ratio:.2f} x", flagged),
    ]
    figures_html = "".join(f'<div class="cf-fig"><div class="l">{html.escape(label)}</div>'
                           f'<div class="v{" hot" if hot else ""}">{html.escape(value)}</div></div>'
                           for label, value, hot in figures)

    rows = [
        ("Predicted status", "Anomaly - flagged for review" if flagged else "Normal - not flagged"),
        ("Known fraud label", case["known_label"]),
        ("Model used", case["source_model"]),
        ("Dataset", dataset_text),
        ("Dataset row reference", f"Row {case['row_number']:,} of the loaded file"),
        ("Transaction time", case["time"]),
    ]
    if case["row_origin"]:
        rows.append(("Row origin", case["row_origin"]))
    rows_html = "".join(f'<tr><td class="k">{html.escape(k)}</td><td class="v">{html.escape(str(v))}</td></tr>'
                        for k, v in rows)

    return (
        '<div class="cf">'
        '<div class="cf-head"><div>'
        '<div class="cf-kicker">The Fraud Bureau / Transaction case file</div>'
        f'<div class="cf-ref">CASE #{html.escape(case["reference"])}</div>'
        f'<div class="cf-meta">{html.escape(case["source_model"])} &nbsp;/&nbsp; {html.escape(dataset_text)}</div>'
        '</div>'
        f'<div class="cf-badge {"flag" if flagged else "normal"}">{html.escape(case["status"])}</div>'
        '</div>'
        f'<div class="cf-figs">{figures_html}</div>'
        '<div class="cf-gauge">'
        '<div class="cf-fig" style="padding:0"><div class="l">Reconstruction error compared with the threshold</div></div>'
        f'<div class="bar"><div class="fill" style="width:{fill_width:.2f}%; background:{fill_colour}"></div>'
        f'<div class="tick" style="left:calc({tick_left:.2f}% - 1px)"></div></div>'
        f'<div class="legend"><span>0</span><span>white line = threshold ({case["threshold"]:.4f})</span>'
        f'<span>error {case["reconstruction_error"]:.4f}</span></div>'
        '</div>'
        f'<table>{rows_html}</table>'
        '<div class="cf-note"><div class="k">Investigation note</div>'
        f'{html.escape(case["note"])} A flagged transaction is not confirmed fraud.</div>'
        '</div>'
    )


# ============================================================
# CHART STYLE (Altair / Vega-Lite, rendered live in the browser)
# ============================================================

def styled(chart, height):
    return (
        chart.properties(height=height)
        .configure(background="transparent", font=FONT, padding={"left": 2, "right": 10, "top": 6, "bottom": 2})
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor=TEXT_2, titleColor=MUTED, labelFontSize=11, titleFontSize=11,
                        titleFontWeight=500, gridColor=GRID, domainColor=BORDER, tickColor=BORDER,
                        labelPadding=6, titlePadding=10, labelFont=FONT, titleFont=FONT)
        .configure_axisX(grid=False)
        .configure_legend(labelColor=TEXT_2, titleColor=MUTED, labelFontSize=11.5, titleFontSize=11,
                          symbolType="square", symbolSize=90, columnPadding=18, labelFont=FONT, titleFont=FONT)
        .configure_text(font=FONT)
    )


def show_chart(chart, height=280):
    st.altair_chart(styled(chart, height), width="stretch", theme=None)


def symlog_ticks(maximum):
    return [v for v in [0, 10, 100, 1_000, 10_000, 100_000, 1_000_000] if v <= maximum * 1.5]


# Compact count labels (10, 100, 1k, 10k, 1M) for compressed count axes.
COUNT_LABELS = ("datum.value >= 1000000 ? format(datum.value / 1000000, ',') + 'M' : "
                "datum.value >= 1000 ? format(datum.value / 1000, ',') + 'k' : format(datum.value, ',')")


def count_axis(maximum, title):
    return alt.Axis(grid=True, values=symlog_ticks(maximum), labelExpr=COUNT_LABELS, title=title)


def log_ticks(low, high):
    """Powers of ten between low and high, for uncluttered log axes."""
    return [10.0 ** e for e in range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)
            if low <= 10.0 ** e <= high]


def donut_chart(frame, colours, centre_value, centre_label):
    """Share of a whole: frame has columns label, value, share."""
    base = alt.Chart(frame).encode(
        theta=alt.Theta("value:Q", stack=True),
        order=alt.Order("order:Q"),
    )
    arcs = base.mark_arc(innerRadius=66, outerRadius=96, stroke=PANEL, strokeWidth=2).encode(
        color=alt.Color("label:N", scale=alt.Scale(domain=list(frame["label"]), range=colours),
                        legend=alt.Legend(orient="bottom", title=None, direction="vertical")),
        tooltip=[alt.Tooltip("label:N", title="Group"), alt.Tooltip("value:Q", title="Transactions", format=","),
                 alt.Tooltip("share:Q", title="Share", format=".3%")],
    )
    value = alt.Chart(pd.DataFrame({"t": [centre_value]})).mark_text(
        fontSize=22, fontWeight=600, color=TEXT, dy=-6).encode(text="t:N")
    label = alt.Chart(pd.DataFrame({"t": [centre_label]})).mark_text(
        fontSize=11, color=TEXT_2, dy=15).encode(text="t:N")
    return arcs + value + label


def threshold_layers(threshold, label_prefix="Threshold"):
    rule = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(
        color=TEXT, strokeDash=[5, 4], strokeWidth=1.5).encode(
        x="threshold:Q", tooltip=[alt.Tooltip("threshold:Q", title=label_prefix, format=".4f")])
    text = alt.Chart(pd.DataFrame({"threshold": [threshold], "label": [f"{label_prefix} {threshold:.4f}"]})).mark_text(
        align="left", dx=6, fontSize=11, color=TEXT, fontWeight=500).encode(
        x="threshold:Q", y=alt.value(8), text="label:N")
    return rule + text


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_resource(show_spinner="Loading the selected model, scaler and threshold...")
def get_model_components(model_key):
    """One cached entry per model. Errors are NOT cached."""
    return load_model_components(*model_file_paths(model_key))


@st.cache_data(show_spinner="Reading the selected dataset...", max_entries=2)
def read_dataset(path_text, identity):
    # `identity` (size + modification time) refreshes the cache if the file changes.
    return pd.read_csv(path_text)


@st.cache_data(show_spinner="Reading the uploaded file...", max_entries=2)
def read_uploaded_dataset(identity, _uploaded_file):
    _uploaded_file.seek(0)
    return pd.read_csv(_uploaded_file)


@st.cache_data
def official_comparison(baseline_path, improved_path, paired_path):
    return load_official_comparison(baseline_path, improved_path, paired_path)


@st.cache_data
def read_csv_cached(path_text):
    return pd.read_csv(path_text)


@st.cache_data
def synthetic_metadata():
    return read_synthetic_metadata()


@st.cache_data
def overview_data(identity, synthetic_start, _df):
    return (dataset_summary(_df), amount_histogram(_df, synthetic_start),
            class_distribution(_df, synthetic_start), amount_statistics(_df, synthetic_start))


@st.cache_data
def training_split_counts(test_index_path, identity, n_original, _df):
    """
    Counts for the ORIGINAL rows of the loaded file (the first n_original rows),
    using the saved official test split.
    """
    original_part = _df.iloc[:n_original]
    test_index = pd.read_csv(test_index_path)["index"].astype(int)
    in_training_split = ~original_part.index.isin(test_index)
    return {
        "test_rows": int((~in_training_split).sum()),
        "training_split_rows": int(in_training_split.sum()),
        "training_legitimate": int((in_training_split & (original_part["Class"] == 0)).sum()),
        "training_fraud": int((in_training_split & (original_part["Class"] == 1)).sum()),
    }


@st.cache_data(show_spinner="Preparing the stored evaluation charts...")
def report_chart_data(baseline_scores_path, improved_scores_path):
    """Reshape the stored official test scores for plotting (nothing is re-scored)."""
    baseline = pd.read_csv(baseline_scores_path)
    improved = pd.read_csv(improved_scores_path)
    pr_parts, hist_parts = [], []
    edges = shared_log_edges([baseline["reconstruction_error"], improved["reconstruction_error"]])
    for name, scores in [("Baseline", baseline), ("Improved v2", improved)]:
        pr_parts.append(pr_curve_points(scores["actual_class"], scores["reconstruction_error"]).assign(model=name))
        hist_parts.append(class_error_histogram(scores["reconstruction_error"], scores["actual_class"], edges)
                          .assign(model=name))
    return pd.concat(pr_parts, ignore_index=True), pd.concat(hist_parts, ignore_index=True)


@st.cache_data
def results_charts(identity, model_key, threshold, _results):
    return log_histogram(_results["reconstruction_error"], threshold), \
        anomaly_score_bands(_results["reconstruction_error"], threshold)


# ============================================================
# BRAND BAR (status chips are filled in at the end, from the real state)
# ============================================================

header_slot = st.empty()

# ============================================================
# SIDEBAR: WORKSPACE
# ============================================================

with st.sidebar:

    st.markdown('<div class="fb-side-brand">THE FRAUD BUREAU <span>/ IntelliFraud</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="fb-side-k" style="margin-top:18px">Workspace</div>', unsafe_allow_html=True)

    dataset_key = st.radio(
        "Dataset",
        options=list(DATASET_OPTIONS),
        format_func=lambda key: DATASET_OPTIONS[key]["label"],
        key="dataset_choice",
        help="An uploaded CSV (Dataset Intelligence tab) overrides this choice.",
    )

    model_key = st.radio(
        "Detection model",
        options=list(MODEL_OPTIONS),
        format_func=lambda key: MODEL_OPTIONS[key]["label"],
        key="model_choice",
        help="Each model uses its own scaler and threshold.",
    )

    st.markdown('<div class="fb-side-k" style="margin-top:6px">Status</div>', unsafe_allow_html=True)
    status_slot = st.empty()  # filled at the end of the script

    with st.expander("Glossary"):
        st.write(
            "**Reconstruction error:** how badly the Autoencoder can rebuild a transaction. "
            "Normal transactions are rebuilt well (small error); unusual ones badly (large error)."
        )
        st.write(
            "**Anomaly threshold:** the error level above which a transaction is flagged. It was set so that "
            "about 5% of legitimate transactions exceed it."
        )
        st.write(
            "**Anomaly score:** reconstruction error divided by the threshold. Above 1 means flagged."
        )
        st.write(
            "**Anomaly:** a transaction above the threshold. It looks unusual and needs review - "
            "it is **not** confirmed fraud."
        )
        st.write(
            "**Known fraud label:** the `Class` column in a labelled file (1 = fraud). It is never given to "
            "the model; it is only compared with the predictions afterwards."
        )

model_option = MODEL_OPTIONS[model_key]
dataset_option = DATASET_OPTIONS[dataset_key]

# ============================================================
# LOAD THE SELECTED MODEL (no silent defaults)
# ============================================================

components = None
model_error = None

try:
    components = get_model_components(model_key)
except Exception as error:  # shown to the user; detection is disabled
    model_error = str(error)

if components is None:
    with st.sidebar:
        st.error("Model not loaded")

if model_error:
    st.error(
        f"**The saved model ({model_option['label']}) could not be loaded, so anomaly detection is disabled.**\n\n"
        f"{model_error}\n\n"
        "Check that the model, scaler and threshold files for this model exist (baseline: `models/`, "
        "improved v2: `models/improved/`), then restart the dashboard."
    )

# key + on_change="rerun" make Streamlit remember the active tab, so clicking
# "Detect Anomalies" does not jump back to the first tab.
tab_dataset, tab_investigations, tab_report = st.tabs(
    ["Dataset Intelligence", "Transaction Investigations", "Bureau Intelligence Report"],
    key="main_tab",
    on_change="rerun",
)

# ============================================================
# TAB 1 - DATASET INTELLIGENCE
# ============================================================

df = None
dataset_id = None
dataset_label = None
dataset_source = None          # "original", "synthetic" or "upload"
synthetic_start = None
metadata = synthetic_metadata() if dataset_key == "synthetic" else None

with tab_dataset:

    heading("Dataset Intelligence",
            "A live overview of the dataset selected in the sidebar, or of your own uploaded CSV. "
            "Every figure and chart below is computed from the loaded file.")

    uploaded_file = st.file_uploader(
        "Upload your own CSV (optional - it overrides the dataset selected in the sidebar)",
        type=["csv"],
    )

    if uploaded_file is not None:

        dataset_id = dataset_identity(uploaded_file, dataset_option["path"])
        dataset_label = f"uploaded file '{uploaded_file.name}'"
        dataset_source = "upload"
        try:
            df = read_uploaded_dataset(dataset_id, uploaded_file)
        except Exception as error:
            st.error(f"The uploaded file could not be read as a CSV: {error}")
        else:
            st.info(
                f"**Uploaded dataset in use:** `{uploaded_file.name}` overrides the "
                f"{dataset_option['label']} selected in the sidebar. Remove the file to switch back."
            )

    elif dataset_option["path"].exists():

        dataset_id = dataset_identity(None, dataset_option["path"])
        dataset_label = f"{dataset_option['label']} ({display_path(dataset_option['path'])})"
        dataset_source = dataset_key
        df = read_dataset(str(dataset_option["path"]), dataset_id)
        if dataset_key == "synthetic":
            synthetic_start = synthetic_start_row(metadata) if metadata else None

    else:

        st.error(
            f"**Dataset file not found:** `{display_path(dataset_option['path'])}`.\n\n"
            f"{dataset_option['how_to_get']}\n\n"
            "Datasets are not stored on GitHub, so they must be added to the `data/` folder on each computer."
        )

    if dataset_source == "synthetic":
        counts = metadata["counts"] if metadata else None
        st.warning(
            "**Synthetic data: this dataset contains artificially generated fraud examples.** "
            + (f"{counts['synthetic_fraud']:,} of its {counts['final_fraud']:,} fraud rows were generated by "
               f"slightly perturbing the {counts['original_fraud']} real fraud transactions; the other "
               f"{counts['original_rows']:,} rows are the original dataset, unchanged. "
               if counts else "")
            + "Use it to demonstrate the dashboard and explore class imbalance only. It is **not** an "
              "independent real-world test dataset and must not be used to claim real-world performance."
        )

    if df is not None:

        summary, histogram_data, distribution, amount_stats = overview_data(dataset_id, synthetic_start, df)

        st.caption(f"Source: {dataset_label}. The Class column (if present) is never used by the model; "
                   "it is shown here only as information about the file.")

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Transactions", f"{summary['total']:,}")
        col2.metric("Legitimate", f"{summary['legitimate']:,}" if summary["has_labels"] else "N/A")
        col3.metric("Known Fraud", f"{summary['fraud']:,}" if summary["has_labels"] else "N/A")
        col4.metric("Fraud Percentage", f"{summary['fraud_percentage']:.3f}%" if summary["has_labels"] else "N/A")
        col5.metric("Missing Values", f"{summary['missing_values']:,}")
        if components is not None:
            required = list(components[1].feature_names_in_)
            present = sum(column in df.columns for column in required)
            col6.metric("Model Input Columns", f"{present} of {len(required)}")
        else:
            col6.metric("Columns", summary["columns"])

        if summary["has_labels"] and len(df):

            colours = [GROUP_COLOURS[g] for g in distribution["group"]]
            chart_col1, chart_col2 = st.columns([3, 2])

            with chart_col1, panel("class_counts"):
                card_header("Transaction distribution by class", "log scale",
                            "Number of transactions in each class. A log scale keeps the small fraud group "
                            "visible next to the legitimate transactions.")
                bars_data = distribution.assign(
                    zero=0, label=[f"{t:,}   {s:.3%}" for t, s in zip(distribution["transactions"],
                                                                     distribution["share"])])
                top = float(bars_data["transactions"].max())
                base = alt.Chart(bars_data).encode(
                    y=alt.Y("group:N", sort=None, title=None,
                            axis=alt.Axis(domain=False, ticks=False, labelLimit=170, labelFontSize=12,
                                          labelColor=TEXT)))
                bars = base.mark_bar(height=24, cornerRadiusEnd=3).encode(
                    x=alt.X("transactions:Q", title="Transactions (log scale)",
                            scale=alt.Scale(type="symlog", domain=[0, top * 120], nice=False),
                            axis=count_axis(top, "Transactions (log scale)")),
                    x2="zero:Q",
                    color=alt.Color("group:N", scale=alt.Scale(domain=list(distribution["group"]), range=colours),
                                    legend=None),
                    tooltip=[alt.Tooltip("group:N", title="Group"),
                             alt.Tooltip("transactions:Q", title="Transactions", format=","),
                             alt.Tooltip("share:Q", title="Share of file", format=".3%")],
                )
                labels = base.mark_text(align="left", dx=8, fontSize=11.5, color=TEXT).encode(
                    x="transactions:Q", text="label:N")
                show_chart(bars + labels, height=74 * len(bars_data) + 40)

            with chart_col2, panel("class_share"):
                card_header("Fraud vs legitimate share", "of all transactions")
                donut_data = distribution.rename(columns={"group": "label", "transactions": "value"})
                donut_data["order"] = range(len(donut_data))
                show_chart(donut_chart(donut_data, colours, f"{summary['fraud_percentage']:.3f}%", "known fraud"),
                           height=74 * len(bars_data) + 40 + 24)

        if not histogram_data.empty:

            chart_col1, chart_col2 = st.columns([1.35, 1])
            groups = histogram_data["group"].unique().tolist()

            with chart_col1, panel("amounts"):
                card_header("Transaction amount distribution by class", "each group scaled to 100%",
                            "Share of each group's transactions per amount range. Groups are scaled "
                            "separately, so the small fraud group can be compared with legitimate transactions.")
                tick_values = [0.0] + [float(np.log10(v + 1)) for v in [1, 10, 100, 1_000, 10_000]]
                tick_values = [v for v in tick_values if v <= histogram_data["bin_end"].max() + 1e-9]
                amount_chart = alt.Chart(histogram_data).mark_line(interpolate="step-after", strokeWidth=2).encode(
                    x=alt.X("bin_start:Q", title="Transaction amount (log scale)",
                            scale=alt.Scale(nice=False),
                            axis=alt.Axis(values=tick_values,
                                          labelExpr="format(round(pow(10, datum.value) - 1), ',')")),
                    y=alt.Y("share:Q", title="Share of the group", axis=alt.Axis(format="%", grid=True)),
                    color=alt.Color("group:N", title=None,
                                    scale=alt.Scale(domain=groups, range=[GROUP_COLOURS[g] for g in groups]),
                                    legend=alt.Legend(orient="top", symbolType="stroke", symbolStrokeWidth=2)),
                    tooltip=[alt.Tooltip("group:N", title="Group"),
                             alt.Tooltip("amount_from:Q", title="Amount from", format=",.2f"),
                             alt.Tooltip("amount_to:Q", title="Amount to", format=",.2f"),
                             alt.Tooltip("share:Q", title="Share of group", format=".2%")],
                ).transform_calculate(amount_from="pow(10, datum.bin_start) - 1",
                                      amount_to="pow(10, datum.bin_end) - 1")
                show_chart(amount_chart, height=280)

            with chart_col2, panel("amount_stats"):
                card_header("Amount statistics by class", "from the loaded file")
                stats_view = amount_stats.set_index("Group").T
                stats_view = stats_view.apply(lambda column: [
                    f"{value:,.0f}" if statistic == "Transactions" else f"{value:,.2f}"
                    for statistic, value in column.items()])
                st.dataframe(stats_view, width="stretch")
                st.caption("Amounts are in the dataset's original currency units. The Amount column is one of "
                           "the 30 model inputs.")

        with panel("preview"):
            card_header("Dataset preview", f"first 10 of {summary['total']:,} rows")
            st.dataframe(df.head(10), width="stretch", height=250)

# ============================================================
# TAB 2 - TRANSACTION INVESTIGATIONS (whole loaded dataset)
# ============================================================

with tab_investigations:

    heading("Transaction Investigations",
            "Score every transaction in the loaded file with the selected model, review the live results, "
            "and open case files for the flagged transactions.")

    threshold_text = f"{components[2]:.4f}" if components is not None else "unavailable"

    with panel("run"):
        run_col1, run_col2 = st.columns([3, 1], vertical_alignment="center")
        with run_col1:
            source_name = ("Uploaded CSV" if dataset_source == "upload" else dataset_option["label"]) \
                if df is not None else "No dataset loaded"
            rows_text = f"{len(df):,} rows" if df is not None else "-"
            st.markdown(
                f'<div class="fb-card-h"><span class="t">{html.escape(model_option["label"])}</span></div>'
                f'<div class="fb-card-d">Dataset: <b>{html.escape(source_name)}</b> ({rows_text}) '
                f'&nbsp;/&nbsp; Threshold: <b>{threshold_text}</b> &nbsp;/&nbsp; '
                f'{html.escape(model_option["summary"])}</div>',
                unsafe_allow_html=True,
            )
        with run_col2:
            detect_clicked = st.button(
                "Detect Anomalies",
                type="primary",
                width="stretch",
                disabled=components is None or df is None,
                key="detect_button",
            )

    st.info(
        f"This scores **every row** in the loaded file with the **{model_option['label']}**. "
        f"An **anomaly** is a transaction whose reconstruction error is above the "
        f"threshold ({threshold_text}): it looks unusual, which is **not** the same as "
        f"confirmed fraud. These counts are **not** a measure of model accuracy."
    )

    # ------------------------------------------------------------
    # Which rows did the selected model learn from? (model-specific)
    # ------------------------------------------------------------
    if (df is not None and dataset_source in ("original", "synthetic") and TEST_INDEX_PATH.exists()
            and "Class" in df.columns):

        n_original = synthetic_start if synthetic_start is not None else len(df)
        split = training_split_counts(str(TEST_INDEX_PATH), dataset_id, n_original, df)
        prefix = ("This is the dataset the model was built from. " if dataset_source == "original" else
                  f"The first {n_original:,} rows of this file are the original dataset the models were built from. ")
        synthetic_note = (
            f" The remaining {len(df) - n_original:,} rows are synthetic fraud generated from the real fraud "
            f"rows - including those in the official test set - so they are not independent new examples."
            if dataset_source == "synthetic" else ""
        )

        if model_key == "baseline":
            st.warning(
                f"{prefix}"
                f"{split['training_split_rows']:,} of these rows (about 80%) belong to the training split. "
                f"Of those, only the **{split['training_legitimate']:,} legitimate transactions were used to "
                f"train the Autoencoder and set its threshold**; the {split['training_fraud']:,} fraudulent "
                f"training-split transactions were excluded from training. Because the threshold is the "
                f"95th percentile of those {split['training_legitimate']:,} transactions' errors, it flags "
                f"about 5% of them by design. For real performance on unseen data, see the "
                f"**Bureau Intelligence Report** tab ({split['test_rows']:,} held-out test transactions)."
                f"{synthetic_note}"
            )
        else:
            counts = improved_training_counts() or {}
            st.warning(
                f"{prefix}"
                f"{split['training_split_rows']:,} of these rows (about 80%) belong to the training split. "
                f"Improved v2 was trained on **{counts.get('training_rows', 0):,} of the legitimate ones** "
                f"(after removing repeated copies and copies of test transactions), and its threshold was set "
                f"on **{counts.get('validation_rows', 0):,} other legitimate rows** (validation set); no fraud "
                f"rows were used. Because the threshold is the 95th percentile of the validation errors, "
                f"about 5% of legitimate rows like these are flagged by design. For real performance on unseen "
                f"data, see the **Bureau Intelligence Report** tab ({split['test_rows']:,} held-out test "
                f"transactions).{synthetic_note}"
            )

    if df is not None:
        st.caption(
            "Note: the `Time` column is 'seconds since the first transaction of that file', so "
            "values in a different file are not directly comparable with the training data."
        )
    else:
        st.info("Load a dataset (sidebar or Dataset Intelligence tab) to run anomaly detection.")

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
                    "model_key": model_key,
                    "results": results,
                    "ignored_columns": ignored_columns,
                    "threshold": components[2],
                }

                st.success("Anomaly detection completed.")

            except InputDataError as error:

                st.error(f"This file cannot be scored: {error}")

            except Exception as error:

                st.error(f"Unexpected error during anomaly detection: {error}")

    # ------------------------------------------------------------
    # Only show results for the dataset AND model selected right now
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
    elif detection is not None and detection["model_key"] != model_key:
        st.session_state.pop("detection", None)
        detection = None
        st.info(
            "A different model is selected now, so the previous results were cleared. "
            "Click **Detect Anomalies** to analyse this dataset with the selected model."
        )

    results = detection["results"] if detection is not None else None

    if results is not None:

        threshold = detection["threshold"]

        heading(f"Detection results / {model_option['label']}",
                "Live charts from the detection results above. They update when the model or dataset changes.")

        if detection["ignored_columns"]:
            st.info(
                "These extra columns were ignored by the model: "
                + ", ".join(f"`{c}`" for c in detection["ignored_columns"])
                + ". Old result columns are recalculated."
            )

        total_transactions = len(results)
        anomaly_count = int(results["anomaly"].sum())
        normal_count = total_transactions - anomaly_count
        anomaly_rate = anomaly_count / total_transactions * 100 if total_transactions else 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Transactions Scored", f"{total_transactions:,}")
        col2.metric("Anomalies Detected", f"{anomaly_count:,}")
        col3.metric("Percentage Flagged", f"{anomaly_rate:.2f}%")
        col4.metric("Model Threshold", f"{threshold:.4f}")

        histogram, bands = results_charts(detection["dataset_id"], detection["model_key"], threshold, results)

        chart_col1, chart_col2 = st.columns([2, 1])

        with chart_col1, panel("error_distribution"):
            card_header("Reconstruction error distribution", "all transactions",
                        f"Every one of the {total_transactions:,} transactions is included - nothing is cut off. "
                        "Bars to the right of the dashed threshold line are anomalies. Both axes are compressed "
                        "so that small bars stay visible.")
            histogram = histogram.assign(baseline=0)
            bars = alt.Chart(histogram).mark_bar(stroke=PANEL, strokeWidth=0.5).encode(
                x=alt.X("bin_start:Q", scale=alt.Scale(type="log"), title="Reconstruction error (log scale)",
                        axis=alt.Axis(format="~g", values=log_ticks(histogram["bin_start"].min(),
                                                                    histogram["bin_end"].max()))),
                x2="bin_end:Q",
                # Bars are drawn from zero (no stacking): only the one bin that
                # contains the threshold can hold both groups.
                y=alt.Y("count:Q", stack=None, scale=alt.Scale(type="symlog", domainMin=0, nice=False),
                        axis=count_axis(histogram["count"].max(), "Transactions (compressed scale)")),
                y2="baseline:Q",
                color=alt.Color("group:N", title=None, scale=alt.Scale(domain=RESULT_DOMAIN, range=RESULT_RANGE),
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("bin_start:Q", title="Error from", format=".4f"),
                         alt.Tooltip("bin_end:Q", title="Error to", format=".4f"),
                         alt.Tooltip("group:N", title="Result"),
                         alt.Tooltip("count:Q", title="Transactions", format=",")],
            )
            show_chart(bars + threshold_layers(threshold), height=300)

        with chart_col2, panel("flag_share"):
            card_header("Flagged vs not flagged", f"{total_transactions:,} scored")
            share = pd.DataFrame({"label": RESULT_DOMAIN, "value": [normal_count, anomaly_count], "order": [0, 1]})
            share["share"] = share["value"] / max(total_transactions, 1)
            show_chart(donut_chart(share, RESULT_RANGE, f"{anomaly_rate:.2f}%", "flagged"), height=316)

        with panel("score_bands"):
            card_header("Anomaly score distribution", "anomaly score = reconstruction error / threshold",
                        "How far each transaction is from the threshold. A score above 1x is flagged; the higher "
                        "the score, the more unusual the transaction looks to the model.")
            bands = bands.assign(baseline=0, label=[f"{c:,}" for c in bands["transactions"]])
            top = float(max(bands["transactions"].max(), 1))
            base = alt.Chart(bands).encode(
                x=alt.X("band:N", sort=list(bands["band"]), title="Anomaly score band",
                        axis=alt.Axis(labelAngle=0, labelColor=TEXT_2)))
            band_bars = base.mark_bar(size=46, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                y=alt.Y("transactions:Q", stack=None, title="Transactions (compressed scale)",
                        scale=alt.Scale(type="symlog", domain=[0, top * 4], nice=False),
                        axis=count_axis(top, "Transactions (compressed scale)")),
                y2="baseline:Q",
                color=alt.Color("result:N", title=None, scale=alt.Scale(domain=RESULT_DOMAIN, range=RESULT_RANGE),
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("band:N", title="Score band"), alt.Tooltip("result:N", title="Result"),
                         alt.Tooltip("transactions:Q", title="Transactions", format=","),
                         alt.Tooltip("share:Q", title="Share of file", format=".3%")],
            )
            band_labels = base.mark_text(dy=-8, fontSize=11.5, color=TEXT).encode(
                y=alt.Y("transactions:Q", stack=None), text="label:N")
            show_chart(band_bars + band_labels, height=250)

    # ------------------------------------------------------------
    # Investigation views (always present; filled after detection)
    # ------------------------------------------------------------
    heading("Investigation workspace")
    inner_cases, inner_anomalies, inner_all, inner_labels = st.tabs(
        ["Transaction Case Files", "Anomalies Detected", "All Scored Transactions", "Predicted vs Known Label"],
        key="investigation_tab",
        on_change="rerun",
    )

    with inner_cases:
        if results is None:
            st.info("Run **Detect Anomalies** to open case files for flagged transactions.")
        else:
            positions = flagged_positions(results)
            if len(positions) == 0:
                st.info("No anomalies were detected, so there are no case files.")
            else:
                shown = positions[:CASE_FILE_CHOICES]
                errors = results["reconstruction_error"].to_numpy()
                amounts = results["Amount"].to_numpy() if "Amount" in results.columns else None
                pick_col, file_col = st.columns([1, 2], gap="large")
                with pick_col:
                    chosen = st.selectbox(
                        "Open case file",
                        options=[int(p) for p in shown],
                        format_func=lambda p: (
                            f"IF-{int(p):06d}  /  error {errors[p]:.2f}"
                            + (f"  /  amount {amounts[p]:,.2f}" if amounts is not None else "")
                        ),
                        key="case_file_choice",
                    )
                    st.caption(
                        f"{len(positions):,} transactions were flagged. The {len(shown):,} with the highest "
                        "reconstruction error can be opened here, highest first. Case references are generated "
                        "from the row number in the loaded file for these detection results only; they are not "
                        "stored records."
                    )
                    context = case_context(results, chosen)
                    below = context["share_below"]
                    below_text = f"{below:.4%}" if below >= 0.9999 else f"{below:.2%}"
                    note(
                        '<span class="k">Case context</span>'
                        f'Rank <b>{context["flagged_rank"]:,}</b> of <b>{context["flagged_total"]:,}</b> flagged '
                        f'transactions by reconstruction error. Its error is higher than '
                        f'<b>{below_text}</b> of all {len(results):,} scored transactions.'
                    )
                with file_col:
                    case = build_case_file(results, chosen, threshold, model_option["label"], synthetic_start)
                    source_text = ("Uploaded CSV" if dataset_source == "upload" else dataset_option["label"])
                    st.markdown(case_file_markup(case, source_text), unsafe_allow_html=True)

    with inner_anomalies:
        if results is None:
            st.info("Run **Detect Anomalies** to see the flagged transactions.")
        else:
            anomalies = results[results["anomaly"] == 1].sort_values("reconstruction_error", ascending=False)
            if len(anomalies) > 0:
                st.markdown(
                    f"**{len(anomalies):,} transactions were flagged as anomalies** "
                    "(sorted from highest to lowest reconstruction error)."
                )
                st.caption(
                    "Each of these needs human review. An anomaly is unusual compared with the "
                    "legitimate training transactions - it is not proof of fraud."
                )
                table = anomalies.copy()
                table.insert(0, "case_reference", [f"IF-{int(p):06d}" for p in
                                                   results.index.get_indexer(anomalies.index)])
                table["anomaly_score"] = table["reconstruction_error"] / threshold
                table["flag_reason"] = f"Reconstruction error above threshold ({threshold:.4f})"
                display_columns = [c for c in ["case_reference", "Time", "Amount", "Class", "reconstruction_error",
                                               "anomaly_score", "anomaly", "flag_reason"] if c in table.columns]
                st.dataframe(
                    table[display_columns], width="stretch", hide_index=True, height=380,
                    column_config={"anomaly_score": st.column_config.NumberColumn("anomaly_score (x threshold)",
                                                                                  format="%.2f")},
                )
                st.download_button(
                    label="Download flagged transactions (CSV)",
                    data=lambda: build_download(anomalies, threshold),
                    file_name=f"intellifraud_anomalies_{model_key}.csv",
                    mime="text/csv",
                    key="download_anomalies",
                )
                st.caption(
                    "Contains every original column of the flagged rows. The file is created when you click "
                    "the button, and it can be uploaded again: the model only reads its own 30 input columns."
                )
            else:
                st.info("No anomalies were detected.")

    with inner_all:
        if results is None:
            st.info("Run **Detect Anomalies** to see the scored transactions.")
        else:
            display_columns = [c for c in ["Time", "Amount", "Class", "reconstruction_error", "anomaly"]
                               if c in results.columns]
            st.dataframe(results[display_columns], width="stretch", height=380)
            st.caption("`anomaly` = predicted anomaly (1) or not (0). `Class` = known label, if the file has one.")
            st.download_button(
                label="Download compact full results (CSV)",
                data=lambda: compact_results_csv(results, model_option["label"], threshold),
                file_name=f"intellifraud_results_{model_key}.csv",
                mime="text/csv",
                key="download_full_results",
            )
            st.caption(
                "One row per scored transaction: row number, known Class label (if available), reconstruction "
                "error, predicted anomaly status, threshold and model. The file is created when you click the button."
            )

    with inner_labels:
        if results is None:
            st.info("Run **Detect Anomalies** to compare predictions with known labels.")
        else:
            comparison = label_comparison(results)
            if comparison is None:
                st.info("This file has no usable `Class` column, so there are no known labels to compare with.")
            else:
                st.dataframe(comparison, width="stretch")
                st.warning(
                    "These counts compare predictions with the dataset's known labels. They are not independent "
                    "model evaluation results because the selected dataset may contain transactions used during "
                    "training."
                )
                if dataset_source == "synthetic":
                    st.caption("This file also contains synthetic fraud rows that are perturbed copies of the real "
                               "fraud rows, including the official test frauds.")
                st.caption("Precision and recall are deliberately not calculated here. The fair, held-out results "
                           "are in the Bureau Intelligence Report tab.")

# ============================================================
# TAB 3 - BUREAU INTELLIGENCE REPORT (stored, verified, read only)
# ============================================================

with tab_report:

    heading("Bureau Intelligence Report",
            "The stored, verified evaluation of both models on the official held-out test set. Charts are drawn "
            "live from the stored result files (test-set scores, metrics and training histories); no metric is "
            "recalculated.")

    missing = [p for p in [BASELINE_METRICS_PATH, IMPROVED_METRICS_PATH, PAIRED_COMPARISON_PATH] if not p.exists()]
    table = None

    if missing:

        st.warning(
            "Official result files not found: "
            + ", ".join(f"`{display_path(p)}`" for p in missing)
            + ". They are created by `src/baseline_evaluation.py` and `src/evaluate_improved.py`."
        )

    else:

        table, conclusion = official_comparison(str(BASELINE_METRICS_PATH), str(IMPROVED_METRICS_PATH),
                                                str(PAIRED_COMPARISON_PATH))
        by_key = table.set_index("key")
        test_rows = conclusion["test_rows"]
        test_fraud = conclusion["test_fraud"]

        st.info(
            f"These are the **stored, verified evaluation results** of both models on the same "
            f"**{test_rows:,} held-out test transactions** ({test_rows - test_fraud:,} legitimate, "
            f"{test_fraud} fraud) that neither model trained on. They do **not** change when you upload "
            f"a different dataset, switch datasets or click Detect Anomalies. Nothing here is recalculated, "
            f"and the synthetic dataset is never used for these numbers."
        )

        # ---- Conclusion of the paired bootstrap --------------------------
        if conclusion["supported"]:
            finding, accent = "An improvement in the Improved v2 model is statistically supported.", "green"
            verdict_text = "the whole 95% confidence interval of the difference is above zero."
        else:
            finding, accent = ("An improvement in the Improved v2 model has not been statistically established.",
                               "amber")
            verdict_text = ("the 95% confidence interval of the difference includes zero, so the test data cannot "
                            "tell the two models apart. Improved v2 uses a cleaner, reproducible method, but it is "
                            "not shown to perform better.")
        note(
            '<span class="k">Statistical finding / paired bootstrap on Average Precision</span>'
            f'<div class="fb-finding">{html.escape(finding)}</div>'
            f'Average Precision: baseline <b>{conclusion["ap_baseline"]:.4f}</b>, Improved v2 '
            f'<b>{conclusion["ap_improved"]:.4f}</b>, difference <b>{conclusion["ap_difference"]:+.4f}</b>, '
            f'paired bootstrap 95% CI <b>[{conclusion["ap_low"]:+.4f}, {conclusion["ap_high"]:+.4f}]</b> '
            f'(1,000 stratified resamples, seed 42): {verdict_text}',
            accent=accent,
        )

        # ---- Formatting helpers -------------------------------------------
        def ci_text(low, high, kind):
            if pd.isna(low):
                return "-"
            return f"{format_metric(low, kind)} - {format_metric(high, kind)}"

        def diff_text(value, kind):
            if kind == "pct":
                return f"{value * 100:+.2f} pp"
            if kind == "num":
                return f"{value:+.4f}"
            if kind == "ratio":
                return f"{value:+.1f}"
            return f"{int(round(value)):+,}"

        def paired_text(low, high, kind):
            if pd.isna(low):
                return "-"
            if kind == "pct":
                return f"[{low * 100:+.2f}, {high * 100:+.2f}] pp"
            return f"[{low:+.4f}, {high:+.4f}]"

        # ---- Comparison KPI cards -------------------------------------------
        cards = []
        for key in ["average_precision", "recall", "precision", "false_positive_rate",
                    "false_alarms_per_fraud_caught"]:
            row = by_key.loc[key]
            paired = (f" / paired 95% CI {paired_text(row['paired_low'], row['paired_high'], row['kind'])}"
                      if not pd.isna(row["paired_low"]) else "")
            cards.append(
                f'<div class="fb-kpi"><div class="l">{html.escape(row["metric"])}</div>'
                f'<div class="row"><span class="m"><i style="background:{BLUE}"></i>Baseline</span>'
                f'<span class="v">{format_metric(row["baseline"], row["kind"])}</span></div>'
                f'<div class="row"><span class="m"><i style="background:{GOLD}"></i>Improved v2</span>'
                f'<span class="v">{format_metric(row["improved"], row["kind"])}</span></div>'
                f'<div class="diff">Difference {diff_text(row["difference"], row["kind"])}{paired}</div></div>'
            )
        st.markdown('<div class="fb-kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)

        # ---- Comparison table ---------------------------------------------
        with panel("comparison_table"):
            card_header("Baseline vs Improved v2 - official held-out test set",
                        f"{test_rows:,} transactions / {test_fraud} fraud")
            display = pd.DataFrame({
                "Metric": table["metric"],
                "Baseline": [format_metric(v, k) for v, k in zip(table["baseline"], table["kind"])],
                "Baseline 95% CI": [ci_text(l, h, k) for l, h, k in
                                    zip(table["baseline_low"], table["baseline_high"], table["kind"])],
                "Improved v2": [format_metric(v, k) for v, k in zip(table["improved"], table["kind"])],
                "Improved v2 95% CI": [ci_text(l, h, k) for l, h, k in
                                       zip(table["improved_low"], table["improved_high"], table["kind"])],
                "Difference (v2 - baseline)": [diff_text(v, k) for v, k in zip(table["difference"], table["kind"])],
                "Paired 95% CI of difference": [paired_text(l, h, k) for l, h, k in
                                                zip(table["paired_low"], table["paired_high"], table["kind"])],
            })
            st.dataframe(display, width="stretch", hide_index=True, height="content")
            st.caption(
                f"Thresholds: baseline {conclusion['baseline_threshold']:.4f}, Improved v2 "
                f"{conclusion['improved_threshold']:.4f}. CI = 95% stratified bootstrap confidence interval. "
                "Accuracy is not shown: a model that never flags anything would score 99.83% here."
            )

        model_scale = alt.Scale(domain=MODEL_DOMAIN, range=MODEL_RANGE)

        # ---- Precision-recall curves and confidence intervals ---------------
        scores_available = BASELINE_SCORES_PATH.exists() and IMPROVED_SCORES_PATH.exists()
        if scores_available:
            pr_points, error_hist = report_chart_data(str(BASELINE_SCORES_PATH), str(IMPROVED_SCORES_PATH))
        else:
            st.warning("Stored test-set scores not found (`results/metrics/*_test_scores.csv`), so the "
                       "precision-recall curves and test-set error distributions cannot be drawn.")

        chart_col1, chart_col2 = st.columns(2)

        if scores_available:
            with chart_col1, panel("pr_curves"):
                card_header("Precision-recall curves", "official test set",
                            f"Drawn from the stored test-set scores. Average Precision: baseline "
                            f"{conclusion['ap_baseline']:.4f}, Improved v2 {conclusion['ap_improved']:.4f}. "
                            "Diamonds mark each model's operating point at its threshold.")
                pr_points = pr_points.assign(step=pr_points.groupby("model").cumcount())
                hover = alt.selection_point(fields=["recall"], nearest=True, on="pointerover", empty=False,
                                            clear="pointerout")
                base = alt.Chart(pr_points).encode(
                    x=alt.X("recall:Q", title="Recall (share of fraud caught)", scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format="%", grid=True)),
                    y=alt.Y("precision:Q", title="Precision (share of flags that are fraud)",
                            scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", grid=True)),
                    color=alt.Color("model:N", scale=model_scale, title=None, legend=alt.Legend(orient="top")),
                )
                lines = base.mark_line(strokeWidth=2).encode(order="step:Q")
                points = base.mark_point(filled=True, size=50).encode(
                    opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                    tooltip=[alt.Tooltip("model:N", title="Model"),
                             alt.Tooltip("recall:Q", title="Recall", format=".2%"),
                             alt.Tooltip("precision:Q", title="Precision", format=".2%")],
                ).add_params(hover)
                operating = pd.DataFrame({
                    "model": MODEL_DOMAIN,
                    "recall": [by_key.loc["recall", "baseline"], by_key.loc["recall", "improved"]],
                    "precision": [by_key.loc["precision", "baseline"], by_key.loc["precision", "improved"]],
                    "threshold": [conclusion["baseline_threshold"], conclusion["improved_threshold"]],
                })
                op_points = alt.Chart(operating).mark_point(shape="diamond", filled=True, size=130,
                                                            stroke=PANEL, strokeWidth=1.5, opacity=1).encode(
                    x="recall:Q", y="precision:Q", color=alt.Color("model:N", scale=model_scale, legend=None),
                    tooltip=[alt.Tooltip("model:N", title="Operating point"),
                             alt.Tooltip("threshold:Q", title="Threshold", format=".4f"),
                             alt.Tooltip("recall:Q", title="Recall", format=".2%"),
                             alt.Tooltip("precision:Q", title="Precision", format=".2%")],
                )
                prevalence = test_fraud / test_rows
                chance = alt.Chart(pd.DataFrame({"p": [prevalence],
                                                 "t": [f"Random guessing ({prevalence:.2%})"]}))
                chance_rule = chance.mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="p:Q")
                chance_text = chance.mark_text(align="left", dy=-7, dx=4, fontSize=10.5, color=MUTED).encode(
                    y="p:Q", x=alt.value(0), text="t:N")
                show_chart(chance_rule + chance_text + lines + points + op_points, height=320)

        with chart_col2, panel("ci_chart"):
            card_header("Key metrics with 95% confidence intervals", "stored bootstrap intervals",
                        "Dot = value on the test set, line = 95% confidence interval (1,000 stratified bootstrap "
                        "resamples). Overlapping intervals mean the models cannot be told apart.")
            ci = ci_frame(table)
            order = list(dict.fromkeys(ci["metric"]))
            base = alt.Chart(ci).encode(
                y=alt.Y("metric:N", sort=order, title=None,
                        axis=alt.Axis(domain=False, ticks=False, labelColor=TEXT, labelFontSize=12)),
                yOffset=alt.YOffset("model:N", sort=MODEL_DOMAIN),
                color=alt.Color("model:N", scale=model_scale, title=None, legend=alt.Legend(orient="top")),
            )
            whiskers = base.mark_rule(strokeWidth=2.5).encode(
                x=alt.X("low:Q", title="Value (0 to 1)", scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(grid=True, format=".1f")),
                x2="high:Q")
            dots = base.mark_point(filled=True, size=70, opacity=1, stroke=PANEL, strokeWidth=1).encode(
                x="value:Q",
                tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("metric:N", title="Metric"),
                         alt.Tooltip("value:Q", title="Value", format=".4f"),
                         alt.Tooltip("low:Q", title="95% CI low", format=".4f"),
                         alt.Tooltip("high:Q", title="95% CI high", format=".4f")])
            show_chart(whiskers + dots, height=320)

        # ---- Confusion matrices and paired differences ----------------------
        confusion = confusion_frame(table)
        confusion["kind"] = np.where(confusion["correct"], "Correct", "Error")
        confusion["count_text"] = confusion["count"].map(lambda c: f"{c:,}")

        chart_col1, chart_col2, chart_col3 = st.columns([1, 1, 1.25])
        for column, model_name, threshold_value in [(chart_col1, "Baseline", conclusion["baseline_threshold"]),
                                                    (chart_col2, "Improved v2", conclusion["improved_threshold"])]:
            with column, panel(f"confusion_{model_name.split()[0].lower()}"):
                card_header(f"Confusion matrix / {model_name}", "",
                            f"At the stored threshold {threshold_value:.4f}.")
                cells = confusion[confusion["model"] == model_name]
                base = alt.Chart(cells).encode(
                    x=alt.X("predicted:N", sort=["Not flagged", "Flagged"], title="Predicted",
                            axis=alt.Axis(orient="top", labelAngle=0, domain=False, ticks=False, labelColor=TEXT)),
                    y=alt.Y("actual:N", sort=["Legitimate", "Fraud"], title="Actual",
                            axis=alt.Axis(domain=False, ticks=False, labelColor=TEXT)),
                )
                rect = base.mark_rect(stroke=PANEL, strokeWidth=4, cornerRadius=5).encode(
                    color=alt.Color("kind:N", scale=alt.Scale(domain=["Correct", "Error"],
                                                              range=["#173A2B", "#45191E"]), legend=None),
                    tooltip=[alt.Tooltip("outcome:N", title="Outcome"), alt.Tooltip("actual:N", title="Actual"),
                             alt.Tooltip("predicted:N", title="Predicted"),
                             alt.Tooltip("count:Q", title="Transactions", format=",")])
                value = base.mark_text(fontSize=20, fontWeight=600, color=TEXT, dy=-7).encode(text="count_text:N")
                label = base.mark_text(fontSize=11, color=TEXT_2, dy=14).encode(text="outcome:N")
                show_chart(rect + value + label, height=230)

        with chart_col3, panel("paired_differences"):
            card_header("Paired difference", "Improved v2 - baseline",
                        "Dot = observed difference, line = paired 95% bootstrap CI. Average Precision (white) is "
                        "the pre-agreed primary metric; an improvement needs its whole interval above zero.")
            diffs = paired_difference_frame(table)
            diffs["role"] = np.where(diffs["primary"], "Primary metric", "Secondary metric")
            order = list(diffs["metric"])
            base = alt.Chart(diffs).encode(
                y=alt.Y("metric:N", sort=order, title=None,
                        axis=alt.Axis(domain=False, ticks=False, labelColor=TEXT, labelFontSize=12)))
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=TEXT, strokeDash=[4, 3],
                                                                 opacity=0.7).encode(x="x:Q")
            whiskers = base.mark_rule(strokeWidth=2.5).encode(
                x=alt.X("low:Q", title="Difference (Improved v2 - baseline)", axis=alt.Axis(grid=True, format="+.2f")),
                x2="high:Q",
                color=alt.Color("role:N", scale=alt.Scale(domain=["Primary metric", "Secondary metric"],
                                                          range=[TEXT, "#5D6570"]),
                                title=None, legend=None))
            dots = base.mark_point(filled=True, size=70, opacity=1, stroke=PANEL, strokeWidth=1).encode(
                x="difference:Q",
                color=alt.Color("role:N", scale=alt.Scale(domain=["Primary metric", "Secondary metric"],
                                                          range=[TEXT, "#5D6570"]), legend=None),
                tooltip=[alt.Tooltip("metric:N", title="Metric"),
                         alt.Tooltip("difference:Q", title="Difference", format="+.4f"),
                         alt.Tooltip("low:Q", title="Paired 95% CI low", format="+.4f"),
                         alt.Tooltip("high:Q", title="Paired 95% CI high", format="+.4f")])
            show_chart(zero + whiskers + dots, height=230)

        # ---- Reconstruction error distributions on the test set -------------
        if scores_available:
            chart_col1, chart_col2 = st.columns(2)
            for column, model_name, threshold_value in [
                    (chart_col1, "Baseline", conclusion["baseline_threshold"]),
                    (chart_col2, "Improved v2", conclusion["improved_threshold"])]:
                with column, panel(f"test_errors_{model_name.split()[0].lower()}"):
                    card_header(f"Test-set reconstruction error / {model_name}", "each class = 100%",
                                "Stored test-set errors for legitimate and fraud transactions. Fraud sits further "
                                "right; the overlap left of the threshold is the fraud that is missed.")
                    curve = error_hist[error_hist["model"] == model_name]
                    colour = alt.Color("group:N", title=None,
                                       scale=alt.Scale(domain=["Legitimate", "Fraud"], range=[GREEN, RED]),
                                       legend=alt.Legend(orient="top"))
                    enc = dict(x=alt.X("bin_start:Q", scale=alt.Scale(type="log"),
                                       title="Reconstruction error (log scale)",
                                       axis=alt.Axis(format="~g", values=log_ticks(curve["bin_start"].min(),
                                                                                    curve["bin_end"].max()))),
                               y=alt.Y("share:Q", stack=None, title="Share of the class",
                                       axis=alt.Axis(format="%", grid=True)),
                               color=colour)
                    area = alt.Chart(curve).mark_area(interpolate="step-after", fillOpacity=0.16).encode(**enc)
                    line = alt.Chart(curve).mark_line(interpolate="step-after", strokeWidth=1.8).encode(
                        **enc, tooltip=[alt.Tooltip("group:N", title="Class"),
                                        alt.Tooltip("bin_start:Q", title="Error from", format=".4f"),
                                        alt.Tooltip("bin_end:Q", title="Error to", format=".4f"),
                                        alt.Tooltip("count:Q", title="Transactions", format=","),
                                        alt.Tooltip("share:Q", title="Share of class", format=".2%")])
                    show_chart(area + line + threshold_layers(threshold_value), height=260)

        # ---- Training and validation loss -----------------------------------
        chart_col1, chart_col2 = st.columns(2)
        for column, model_name, path, colour, side in [
                (chart_col1, "Baseline", BASELINE_HISTORY_PATH, BLUE, "10 epochs"),
                (chart_col2, "Improved v2", IMPROVED_HISTORY_PATH, GOLD, "100 epochs, no early stop")]:
            with column, panel(f"loss_{model_name.split()[0].lower()}"):
                card_header(f"Training and validation loss / {model_name}", side)
                if not path.exists():
                    st.info(f"`{display_path(path)}` not found.")
                    continue
                history = training_history(path, model_name)
                loss_chart = alt.Chart(history).mark_line(strokeWidth=2, color=colour).encode(
                    x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1, format="d")),
                    y=alt.Y("loss_value:Q", title="Loss (mean squared error)", scale=alt.Scale(zero=False),
                            axis=alt.Axis(grid=True)),
                    strokeDash=alt.StrokeDash("split:N", title=None,
                                              scale=alt.Scale(domain=["Training loss", "Validation loss"],
                                                              range=[[1, 0], [5, 4]]),
                                              legend=alt.Legend(orient="top", symbolType="stroke",
                                                                symbolStrokeColor=colour)),
                    tooltip=[alt.Tooltip("split:N", title="Series"), alt.Tooltip("epoch:Q", title="Epoch"),
                             alt.Tooltip("loss_value:Q", title="Loss", format=".4f")],
                )
                show_chart(loss_chart, height=240)

        # ---- Why false positives matter ------------------------------------
        fp, tp, per_catch = by_key.loc["false_positives"], by_key.loc["true_positives"], \
            by_key.loc["false_alarms_per_fraud_caught"]
        note(
            '<span class="k">Why false positives matter</span>'
            f'Every flagged transaction creates an alert that an analyst must check, and it may block or delay a '
            f'genuine customer\'s payment. On the test set the baseline raised <b>{int(fp["baseline"]):,}</b> false '
            f'alarms to catch <b>{int(tp["baseline"])}</b> frauds (<b>{per_catch["baseline"]:.1f}</b> per fraud '
            f'caught); Improved v2 raised <b>{int(fp["improved"]):,}</b> for <b>{int(tp["improved"])}</b> '
            f'(<b>{per_catch["improved"]:.1f}</b>). A lower threshold catches more fraud but creates more false '
            f'alarms; a higher one does the opposite. That trade-off - not accuracy - is what matters in fraud '
            f'detection.',
            accent="red",
        )
        st.caption("Full reports: `results/metrics/baseline_evaluation_report.txt` and "
                   "`results/metrics/improved_evaluation_report.txt`.")

    # ---- Earlier 3-model comparison (stored) ---------------------------------
    with st.expander("Earlier comparison: Autoencoder vs Isolation Forest vs One-Class SVM (stored)"):
        if not THREE_MODEL_COMPARISON_PATH.exists():
            st.warning("`models/model_comparison.csv` not found. It is created by `src/compare_models.py`.")
        else:
            comparison_df = read_csv_cached(str(THREE_MODEL_COMPARISON_PATH))
            st.info(
                "Stored results from `src/compare_models.py` for the original baseline Autoencoder, measured on "
                "the same held-out test set. They are **not** recalculated by the dashboard."
            )
            formatted = comparison_df.copy()
            for column in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                formatted[column] = formatted[column].map(lambda x: f"{x:.2%}")
            st.dataframe(formatted, width="stretch", hide_index=True)
            f1_ci = None
            if table is not None:
                f1_row = table.set_index("key").loc["f1_score"]
                f1_ci = (f1_row["baseline_low"], f1_row["baseline_high"])
            st.write(comparison_summary(comparison_df, f1_ci))
            st.caption(
                "One-Class SVM was trained on a random sample of only 3,000 legitimate training transactions "
                "(for speed), while the Autoencoder and Isolation Forest used all of them, so the comparison is "
                "not fully equal."
            )

# ============================================================
# BRAND BAR AND SIDEBAR STATUS - filled last, from the real state
# ============================================================

if components is None:
    readiness = ("fail", "MODEL OFFLINE")
elif df is None:
    readiness = ("warn", "AWAITING DATA")
else:
    readiness = ("ok", "SYSTEM READY")

source_text = ("Uploaded CSV" if dataset_source == "upload" else dataset_option["label"]) if df is not None \
    else "Not loaded"

header_slot.markdown(
    '<div class="fb-top">'
    '<div class="fb-brand"><div class="fb-mark"><span></span></div><div>'
    '<div class="fb-name">THE FRAUD BUREAU<span class="sep">/</span><span class="product">IntelliFraud</span></div>'
    '<div class="fb-sub">Credit Card Fraud Detection Using Deep Learning-Based Autoencoders</div>'
    '</div></div>'
    '<div class="fb-chips">'
    f'<span class="fb-chip fb-status"><span class="fb-dot {readiness[0]}"></span>{readiness[1]}</span>'
    f'<span class="fb-chip"><span class="k">Model</span><b>{html.escape(model_option["label"])}</b></span>'
    f'<span class="fb-chip"><span class="k">Data</span><b>{html.escape(source_text)}</b></span>'
    '</div></div>',
    unsafe_allow_html=True,
)

status_rows = [
    ("Dataset", source_text, "warn" if dataset_source == "synthetic" else ""),
    ("Model", model_option["tag"], ""),
]
if components is not None:
    status_rows += [
        ("Architecture", architecture_text(components[0]), ""),
        ("Threshold", f"{components[2]:.4f}", ""),
        ("Model status", "Loaded", "ok"),
    ]
else:
    status_rows += [("Model status", "Model not loaded", "fail")]
if results is not None:
    status_rows.append(("Detection", f"{int(results['anomaly'].sum()):,} flagged", "ok"))
else:
    status_rows.append(("Detection", "Not run", ""))
status_slot.markdown(status_list_html(status_rows), unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="fb-footer"><span>THE FRAUD BUREAU / IntelliFraud</span>'
    '<span>Flagged transactions are anomalies for review, not confirmed fraud</span></div>',
    unsafe_allow_html=True,
)
