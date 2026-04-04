import streamlit as st
from datetime import datetime
import sys
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="GDP Nowcast Terminal")

# --- 2. PATH FIX ---
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# --- 3. IMPORTS ---
from src.data_preprocessing import (
    aggregate_to_quarterly,
    merge_data,
    transform_series
)
from export_ar_history import build_historical_ar_csv
from export_adl_history import build_historical_adl_csv
from export_bridge_history import build_historical_bridge_csv
from frontend.components.atlanta_fed import annualize_gdp_growth


# --- 4. CUSTOM LOADERS FOR LIVE API FILES ---
def load_and_transform_live_monthly(monthly_path, md_reference_path):
    monthly_raw = pd.read_csv(monthly_path, parse_dates=True, index_col=0)

    md_meta = pd.read_csv(md_reference_path, nrows=1)
    tcodes = md_meta.iloc[0].to_dict()

    transformed_list = []

    for col in monthly_raw.columns:
        if col not in tcodes:
            continue

        code = tcodes[col]
        if pd.isna(code):
            continue

        try:
            s = transform_series(monthly_raw[col].astype(float), int(code))
            s.name = col
            transformed_list.append(s)
        except Exception:
            continue

    MD_trans = pd.concat(transformed_list, axis=1).sort_index()

    vars_to_drop = ['ACOGNO', 'UMCSENTx', 'TWEXAFEGSMTHx', 'ANDENOx', 'VIXCLSx']
    MD_trans = MD_trans.drop(columns=vars_to_drop, errors='ignore')

    return MD_trans


def load_live_quarterly_gdp_annualized(gdp_path):
    gdp_raw = pd.read_csv(gdp_path, parse_dates=True, index_col=0).squeeze()
    gdp_raw.index = pd.to_datetime(gdp_raw.index)
    gdp_raw.index = gdp_raw.index.to_period("Q")

    GDP_growth = (np.log(gdp_raw).diff() * 400).rename("GDP_growth")
    return GDP_growth


# --- 5. LOAD GDP FOR CHARTS / AR ---
gdp_path = ROOT_DIR / "data" / "live_api_quarterly_gdp.csv"
GDP_growth = load_live_quarterly_gdp_annualized(gdp_path)
gdp_data = GDP_growth.copy()


# --- 6. PREPARE MERGED MODEL DATA ---
@st.cache_data
def prepare_data():
    monthly_path = ROOT_DIR / "data" / "live_api_monthly.csv"
    gdp_path = ROOT_DIR / "data" / "live_api_quarterly_gdp.csv"
    md_reference_path = ROOT_DIR / "data" / "2026-02-MD.csv"

    MD_trans = load_and_transform_live_monthly(monthly_path, md_reference_path)
    GDP_growth = load_live_quarterly_gdp_annualized(gdp_path)

    monthly_q = aggregate_to_quarterly(MD_trans)
    if not isinstance(monthly_q.index, pd.PeriodIndex):
        monthly_q.index = monthly_q.index.to_period("Q")

    # keep a few columns needed for ADL + bridge
    required_cols = [
    "DPCERA3M086SBEA",
    "UEMP15T26",
    "DMANEMP",
    "IPDMAT",
    "W875RX1",
    "UNRATE",
    "BAA",
    "AAA",
    "HOUST",
    "PAYEMS",
    "PERMITNE",
    "HWIURATIO",
]
    available_cols = [c for c in required_cols if c in monthly_q.columns]
    monthly_q = monthly_q[available_cols].copy()

    data = monthly_q.join(GDP_growth, how="left")

    data["covid_dummy"] = 0
    data.loc[
        (data.index >= pd.Period("2020Q1", freq="Q")) &
        (data.index <= pd.Period("2020Q4", freq="Q")),
        "covid_dummy"
    ] = 1

    return data, MD_trans, GDP_growth, monthly_q

data, MD_trans, GDP_growth, monthly_q = prepare_data()


# --- 7. BUILD HISTORICAL MODEL OUTPUTS ---
@st.cache_data
def prepare_ar_history(gdp_series):
    output_path = ROOT_DIR / "data" / "historical_gdp_ar_predictions.csv"
    return build_historical_ar_csv(
        gdp_series=gdp_series,
        output_path=output_path,
        max_lag=8,
        min_train_size=20,
    )


@st.cache_data
def prepare_adl_history(data):
    output_path = ROOT_DIR / "data" / "historical_gdp_adl_predictions.csv"
    return build_historical_adl_csv(
        data=data,
        output_path=output_path,
        target_col="GDP_growth",
        min_train_size=20,
    )


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


ar_history_df = prepare_ar_history(gdp_data)
adl_history_df = prepare_adl_history(data)

# bridge_selected_variables = [
#     "IPDMAT",
#     "DPCERA3M086SBEA",
#     "PAYEMS",
#     "UEMP15T26",
#     "PERMITNE",
#     "UNRATE",
#     "HWIURATIO",
# ]
bridge_selected_variables = [
    "IPDMAT",
    "DPCERA3M086SBEA",
    "PAYEMS",
    "UEMP15T26",
    "PERMITNE",
    "UNRATE",
]
bridge_history_df = prepare_bridge_history(data, bridge_selected_variables)


# --- 8. COMPONENT IMPORTS ---
try:
    from frontend.components import (
        config_panel,
        live_metric,
        biz_cycle,
        history_chart,
        intra_quarter_chart,
        live_graph,
    )
except ModuleNotFoundError:
    from components import (
        config_panel,
        live_metric,
        biz_cycle,
        history_chart,
        intra_quarter_chart,
        live_graph,
    )
@st.dialog("Dashboard Update")
def success_popup():
    st.success("✅ Nowcast dashboard updated successfully!")
    st.markdown("The latest FRED data has been downloaded and the GDP models have been re-run.")


# --- 9. PAGE STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }

    [data-testid="stMetric"] {
        background-color: #1e2127;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }

    div.stButton > button {
        height: 114px;
        background-color: #1e2127;
        border: 1px solid #30363d;
        border-radius: 12px;
        color: #A0AAB5;
        font-size: 22px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        border-color: #00FF00;
        color: #00FF00;
        background-color: #1e2127;
    }
    </style>
""", unsafe_allow_html=True)


# --- 10. HEADER ---
col_title, col_status = st.columns([3, 1])

with col_title:
    st.title("GDP Nowcast Terminal")
    st.markdown(
        "DSE3101 | <span style='background-color:#00ff00; padding:2px 8px; border-radius:10px; color: black; font-weight: bold;'>⚡ LIVE FEED</span>",
        unsafe_allow_html=True
    )

with col_status:
    st.write(f"**Last Sync:** {datetime.now().strftime('%H:%M:%S')}")

st.divider()

# Check if the pop-up trigger was set
if st.session_state.get("show_popup", False):
    # Reset the trigger immediately so it doesn't get stuck in an infinite loop
    st.session_state["show_popup"] = False 
    success_popup()

# --- 11. TABS ---
tab1, tab2, tab3 = st.tabs(["Right Now", "Monthly Nowcast", "History Chart"])

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        biz_cycle.render(gdp_data)

    with col2:
        live_metric.render()

    with col3:
        if st.button("Refresh data (Its gonna take like 2mins bro)", use_container_width=True):
            # Show a loading spinner so the user knows it's thinking
            with st.spinner("Downloading FRED Data & Running Nowcast..."):
                try:
                    # 1. Run the FRED Data downloader
                    fred_script = ROOT_DIR / "src"/ "api_preprocessing.py"
                    subprocess.run([sys.executable, str(fred_script)], check=True)
                    
                    # 2. Run the Nowcast Model to generate new predictions
                    model_script = ROOT_DIR / "src"/ "live_nowcast.py"
                    subprocess.run([sys.executable, str(model_script)], check=True)
                    st.session_state["show_popup"] = True         
                              
                    # 3. Clear old memory and reload the page with the new CSVs!
                    st.cache_data.clear()
                    st.rerun()
                    
                except subprocess.CalledProcessError as e:
                    st.error(f"Pipeline failed! Check terminal for details. Error code: {e.returncode}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    st.divider()
    st.container()
    live_graph.render()


with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    intra_quarter_chart.render(gdp_data)


with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 7])

    with col1:
        config_panel.render()

    with col2:
        history_chart.render(gdp_data)
