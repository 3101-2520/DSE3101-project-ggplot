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
from frontend.components.atlanta_fed import get_historical_nowcasts
from frontend.components.atlanta_fed import annualize_gdp_growth

qd_path = ROOT_DIR / "data" / "2026-02-QD.csv"
qd_trans = load_and_transform_qd(str(qd_path), gdp_col="GDPC1")
gdp_data = annualize_gdp_growth(qd_trans)

@st.cache_data
def prepare_data():
    md_path = ROOT_DIR / "data" / "2026-02-MD.csv"
    qd_path = ROOT_DIR / "data" / "2026-02-QD.csv"

    MD_trans = load_and_transform_md(md_path)
    vars_to_drop = ['ACOGNO', 'UMCSENTx', 'TWEXAFEGSMTHx', 'ANDENOx', 'VIXCLSx']
    MD_trans = MD_trans.drop(columns=vars_to_drop, errors='ignore')

    GDP_growth = load_and_transform_qd(qd_path, gdp_col="GDPC1")
    GDP_growth = annualize_gdp_growth(GDP_growth)
    GDP_growth.name = "GDP_growth"

    start_period = pd.Period("1960Q1", freq="Q")
    end_period = pd.Period(pd.Timestamp.now(), freq="Q")
    start_date = start_period.start_time
    end_date = end_period.end_time

    MD_trans = MD_trans.loc[start_date:end_date]
    GDP_growth = GDP_growth.loc[start_period:end_period]

    monthly_q = aggregate_to_quarterly(MD_trans)
    data, X, y = merge_data(monthly_q, GDP_growth)

    data["covid_dummy"] = 0
    data.loc[
        (data.index >= pd.Period("2020Q1", freq="Q")) &
        (data.index <= pd.Period("2020Q2", freq="Q")),
        "covid_dummy"
    ] = 1

    return data, X, y, MD_trans, GDP_growth

data, X, y, MD_trans, GDP_growth = prepare_data()

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
bridge_selected_variables = ["IPDMAT", "DPCERA3M086SBEA", "PAYEMS", "UEMP15T26", "PERMITNE", "UNRATE", "HWIURATIO"]
bridge_history_df = prepare_bridge_history(data, bridge_selected_variables)


# --- 4. COMPONENT IMPORTS ---
try:
    from frontend.components import config_panel, live_metric, biz_cycle, history_chart, intra_quarter_chart, live_graph
except ModuleNotFoundError:
    from components import config_panel, live_metric, biz_cycle, history_chart, intra_quarter_chart, live_graph

# --- 5. PAGE STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    [data-testid="stMetric"] {
        background-color: #1e2127;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    
    /* NEW: Make the Streamlit button look exactly like the metric cards */
    div.stButton > button {
        height: 114px; /* Matches the exact height of your custom HTML cards */
        background-color: #1e2127;
        border: 1px solid #30363d;
        border-radius: 12px;
        color: #A0AAB5;
        font-size: 22px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    /* Make it glow green when hovered! */
    div.stButton > button:hover {
        border-color: #00FF00;
        color: #00FF00;
        background-color: #1e2127;
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

# --- 7. MAIN LAYOUT TABS ---
tab1, tab2, tab3 = st.tabs(["Right Now", "Monthly Nowcast", "History Chart"])

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    # Pass the actual GDP data into the chart!
    intra_quarter_chart.render(gdp_data)

with tab3: 
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 7])
    with col1:
        config_panel.render()
    with col2:
        history_chart.render(gdp_data)

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- THE RULE OF THREE LAYOUT ---
    col1, col2, col3 = st.columns(3) # 3 equal columns!
    
    with col1:
        biz_cycle.render(gdp_data)
        
    with col2:
        live_metric.render()
        
    with col3:
        # The button is now a massive card itself, so no spacer is needed!
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
            
    st.divider()

    # The Live Graph
    st.container()
    live_graph.render()