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

# --- 3. DATA LOADING ---
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
from export_bridge_evolution import build_bridge_evolution_csv 
from frontend.components.atlanta_fed import get_historical_nowcasts

@st.cache_data(ttl=3600)
def get_actual_gdp_from_fred():
    nowcasts_df = get_historical_nowcasts()
    if nowcasts_df.empty or "Real GDP (Actual)" not in nowcasts_df.columns:
        return pd.DataFrame(columns=["Year and Quarter", "Actual GDP growth"])
    actual_df = nowcasts_df[["Real GDP (Actual)"]].reset_index()
    first_col = actual_df.columns[0]
    actual_df = actual_df.rename(columns={first_col: "Year and Quarter", "Real GDP (Actual)": "Actual GDP growth"})
    return actual_df

@st.cache_data
def get_modeling_data():
    md_path = ROOT_DIR / "data" / "2026-02-MD.csv"
    qd_path = ROOT_DIR / "data" / "2026-02-QD.csv"

    MD_trans = load_and_transform_md(str(md_path))
    vars_to_drop = ['ACOGNO', 'UMCSENTx', 'TWEXAFEGSMTHx', 'ANDENOx', 'VIXCLSx']
    MD_trans = MD_trans.drop(columns=vars_to_drop, errors='ignore')

    GDP_growth = load_and_transform_qd(str(qd_path), gdp_col='GDPC1')

    start_period = pd.Period('1960Q1', freq='Q')
    start_date = start_period.start_time

    MD_trans = MD_trans.loc[start_date:]
    GDP_growth = GDP_growth.loc[start_period:]

    # STEP 4 & 5 FIXED: Do not drop NaNs so the live quarter stays!
    monthly_q = aggregate_to_quarterly(MD_trans)
    data = monthly_q.copy()
    actual_gdp_df = get_actual_gdp_from_fred().copy()

    if not actual_gdp_df.empty:
        fred_gdp = actual_gdp_df.rename(columns={"Actual GDP growth": "GDP_growth"}).copy()
        fred_gdp["Year and Quarter"] = fred_gdp["Year and Quarter"].str.replace(" ", "", regex=False)
        fred_gdp["Year and Quarter"] = pd.PeriodIndex(fred_gdp["Year and Quarter"], freq="Q")
        fred_gdp = fred_gdp.set_index("Year and Quarter")[["GDP_growth"]]
        data = data.join(fred_gdp, how="left")
    else:
        data = data.join(GDP_growth, how="left")

    data['covid_dummy'] = 0
    data.loc[
        (data.index >= pd.Period('2020Q1', freq='Q')) &
        (data.index <= pd.Period('2020Q2', freq='Q')), 'covid_dummy'
    ] = 1

    test_size = 8
    # Train data strictly drops the missing GDP rows so the model can fit
    train_data = data.dropna(subset=['GDP_growth']).iloc[:-test_size].copy()

    selected_summary = select_features_rlasso(
        train_data, target_col='GDP_growth', exclude_cols=['covid_dummy']
    )
    selected = list(selected_summary["feature"])

    return data, MD_trans, selected

actual_gdp_df = get_actual_gdp_from_fred()
gdp_data = (
    actual_gdp_df.assign(
        **{"Year and Quarter": lambda df: pd.PeriodIndex(df["Year and Quarter"].astype(str).str.replace(" ", "", regex=False), freq="Q")}
    ).set_index("Year and Quarter")["Actual GDP growth"].sort_index().dropna()
)
data, md_trans, selected = get_modeling_data()

# -- Historical Exporters --
@st.cache_data
def prepare_ar_history(gdp_series):
    output_path = ROOT_DIR / "data" / "historical_gdp_ar_predictions.csv"
    return build_historical_ar_csv(gdp_series=gdp_series, output_path=output_path, max_lag=8, min_train_size=20)
ar_history_df = prepare_ar_history(gdp_data)

@st.cache_data
def prepare_adl_history(data):
    output_path = ROOT_DIR / "data" / "historical_gdp_adl_predictions.csv"
    return build_historical_adl_csv(data=data, output_path=output_path, target_col="GDP_growth", min_train_size=20)
adl_history_df = prepare_adl_history(data)

@st.cache_data
def prepare_bridge_history(data, selected):
    output_path = ROOT_DIR / "data" / "historical_gdp_bridge_predictions.csv"
    return build_historical_bridge_csv(data=data, selected=selected, output_path=output_path, target_col="GDP_growth", min_train_size=20)
bridge_history_df = prepare_bridge_history(data, selected)

# -- Evolution Exporter --
@st.cache_data
def prepare_bridge_evolution(data, selected):
    output_path = ROOT_DIR / "data" / "bridge_evolution.csv"
    return build_bridge_evolution_csv(data=data, selected_names=selected, output_path=output_path, test_size=8)
bridge_evo_df = prepare_bridge_evolution(data, selected)

# --- 4. COMPONENT IMPORTS ---
try:
    from frontend.components import config_panel, live_metric, biz_cycle, history_chart, subscription_ui, intra_quarter_chart
except ModuleNotFoundError:
    from components import config_panel, live_metric, biz_cycle, history_chart, subscription_ui, intra_quarter_chart

# --- 5. PAGE STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    [data-testid="stMetric"] { background-color: #1e2127; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
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
    config_panel.render()

st.container()
history_chart.render(gdp_data)

with top_right:
    col_left, col_right = st.columns([2, 1])
    with col_left:
        live_metric.render() # <-- Empty
    with col_right:
        biz_cycle.render(bridge_history_df) # <-- Has Data
    st.markdown("<br>", unsafe_allow_html=True)
    subscription_ui.render()

st.divider()
st.container()
intra_quarter_chart.render()