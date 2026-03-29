import streamlit as st
from datetime import datetime
import sys
from pathlib import Path
import requests
import certifi
import pandas as pd

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="GDP Nowcast Terminal")

# --- 2. PATH FIX ---
# Ensure we can see 'src' and 'frontend' from the root
ROOT_DIR = Path(__file__).resolve().parents[1] 
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# --- 3. DATA LOADING (Centralized) ---
from src.data_preprocessing import (
    load_and_transform_md,
    load_and_transform_qd,
    aggregate_to_quarterly,
    merge_data,
)
from src.feature_selection import select_features_rlasso

from export_ar_history import build_historical_ar_csv
from export_adl_history import build_historical_adl_csv
from export_bridge_history import build_historical_bridge_csv


@st.cache_data
def get_historical_gdp_series():
    qd_path = ROOT_DIR / "data" / "2026-02-QD.csv"
    return load_and_transform_qd(str(qd_path), gdp_col="GDPC1")


@st.cache_data
def get_modeling_data():
    md_path = ROOT_DIR / "data" / "2026-02-MD.csv"
    qd_path = ROOT_DIR / "data" / "2026-02-QD.csv"

    # Step 1: monthly data
    MD_trans = load_and_transform_md(str(md_path))

    vars_to_drop = ['ACOGNO', 'UMCSENTx', 'TWEXAFEGSMTHx', 'ANDENOx', 'VIXCLSx']
    MD_trans = MD_trans.drop(columns=vars_to_drop, errors='ignore')

    # Step 2: quarterly GDP
    GDP_growth = load_and_transform_qd(str(qd_path), gdp_col='GDPC1')

    # Step 3: same sample filter as execution.py
    start_period = pd.Period('1960Q1', freq='Q')
    start_date = start_period.start_time

    MD_trans = MD_trans.loc[start_date:]
    GDP_growth = GDP_growth.loc[start_period:]

    # Step 4: aggregate + merge
    monthly_q = aggregate_to_quarterly(MD_trans)
    data, X, y = merge_data(monthly_q, GDP_growth)

    # Step 5: add covid dummy
    data['covid_dummy'] = 0
    data.loc[
        (data.index >= pd.Period('2020Q1', freq='Q')) &
        (data.index <= pd.Period('2020Q2', freq='Q')),
        'covid_dummy'
    ] = 1

    # Step 6: use same training split and feature selection
    test_size = 8
    train_data = data.iloc[:-test_size].copy()

    selected_summary = select_features_rlasso(
        train_data,
        target_col='GDP_growth',
        exclude_cols=['covid_dummy']
    )
    selected = list(selected_summary["feature"])

    return data, MD_trans, selected


gdp_data = get_historical_gdp_series()
data, md_trans, selected = get_modeling_data()

# -- AR Data --
@st.cache_data
def prepare_ar_history(gdp_series):
    output_path = ROOT_DIR / "data" / "historical_gdp_ar_predictions.csv"
    return build_historical_ar_csv(
        gdp_series=gdp_series,
        output_path=output_path,
        max_lag=8,
        min_train_size=20,
    )

ar_history_df = prepare_ar_history(gdp_data)

# -- ADL Data --
@st.cache_data
def prepare_adl_history(data):
    output_path = ROOT_DIR / "data" / "historical_gdp_adl_predictions.csv"
    return build_historical_adl_csv(
        data=data,
        output_path=output_path,
        target_col="GDP_growth",
        min_train_size=20,
    )

adl_history_df = prepare_adl_history(data)

# -- Bridge Data --
@st.cache_data
def prepare_bridge_history(data, selected):
    output_path = ROOT_DIR / "data" / "historical_gdp_bridge_predictions.csv"
    return build_historical_bridge_csv(
        data=data,
        selected=selected,
        output_path=output_path,
        target_col="GDP_growth",
        min_train_size=20,
    )

bridge_history_df = prepare_bridge_history(data, selected)


# --- 4. COMPONENT IMPORTS ---
try:
    from frontend.components import config_panel, live_metric, history_chart, subscription_ui
except ModuleNotFoundError:
    from components import config_panel, live_metric, history_chart, subscription_ui

# --- 5. PAGE STYLING ---
#st.set_page_config(layout="wide", page_title="GDP Nowcast Terminal")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; } /* Dark theme for that terminal look */
    [data-testid="stMetric"] {
        background-color: #1e2127;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 6. HEADER ---
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("GDP Nowcast Terminal")
    st.markdown("DSE3101 | <span style='background-color:#00ff00; padding:2px 8px; border-radius:10px; color: black; font-weight: bold;'>⚡ LIVE FEED</span>", unsafe_allow_html=True)

with col_status:
    st.write(f"**Last Sync:** {datetime.now().strftime('%H:%M:%S')}")

st.divider()

# --- 7. MAIN LAYOUT GRID ---
top_left, top_right = st.columns([1, 2.5])

with top_left:
    # Pass gdp_data so it knows which years are available
    config_panel.render()

with top_right:
    subscription_ui.render()
    # Renders the small metric cards at the top
    #live_metric.render()

st.container()
# Pass gdp_data to the main chart for the integrated view
history_chart.render(gdp_data)