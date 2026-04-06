import streamlit as st
from datetime import datetime
import sys
from streamlit_option_menu import option_menu
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
import base64
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.markdown(
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">', 
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Target the main container that holds everything */
    .block-container {
        padding-top: 1rem !important; /* Reduces the gap at the top */
        padding-bottom: 0rem !important; /* Reduces the gap at the bottom */
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
from frontend.export_ar_history import build_historical_ar_csv
from frontend.export_adl_history import build_historical_adl_csv
from frontend.export_bridge_history import build_historical_bridge_csv
from frontend.components.fred_industry_models import annualize_gdp_growth


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

bridge_selected_variables = [
    'DPCERA3M086SBEA', 
    'UEMP15T26', 
    'DMANEMP', 
    'IPDMAT', 
    'W875RX1', 
    'UNRATE']

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
        fred_industry_models,
        fred_nowcast
    )
except ModuleNotFoundError:
    from components import (
        config_panel,
        live_metric,
        biz_cycle,
        history_chart,
        intra_quarter_chart,
        live_graph,
        fred_industry_models,
        fred_nowcast
    )
@st.dialog("Dashboard Update")
def success_popup():
    st.success("✅ Nowcast dashboard updated successfully!")
    st.markdown("The latest FRED data has been downloaded and the GDP models have been re-run.")

# --- 9. PAGE STYLING ---
st.markdown("""
    <style>
    @keyframes pulse-glow {
        0% { opacity: 1; text-shadow: 0 0 5px #00ff00; }
        50% { opacity: 0.6; text-shadow: 0 0 20px #00ff00; }
        100% { opacity: 1; text-shadow: 0 0 5px #00ff00; }
    }

    @keyframes pulse-green {
        0% { opacity: 1; text-shadow: 0 0 5px #00FF00; }
        50% { opacity: 0.5; text-shadow: 0 0 20px #00FF00; }
        100% { opacity: 1; text-shadow: 0 0 5px #00FF00; }
    }

    @keyframes pulse-red {
        0% { opacity: 1; text-shadow: 0 0 5px #FF3333; }
        50% { opacity: 0.5; text-shadow: 0 0 20px #FF3333; }
        100% { opacity: 1; text-shadow: 0 0 5px #FF3333; }
    }

    .flash-text {
        animation: pulse-glow 2s infinite;
        color: #00ff00;
        font-weight: bold;
        font-size: 2rem;
        text-align: center;
        margin: 0;
    }

    .flash-green {
        animation: pulse-green 2s infinite;
    }

    .flash-red {
        animation: pulse-red 2s infinite;
    }

    .custom-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
    }

    .card-label {
        color: #a1a1aa;
        font-size: 0.8rem;
        margin-bottom: 10px;
        text-transform: uppercase;
        text-align: center;
    }

    /* GLOBAL & MAIN AREA */
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    
    /* Hides the collapse button inside the sidebar */
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* METRICS & CARDS */
    [data-testid="stMetric"] {
        background-color: #1e2127;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }

    /* LARGE REFRESH BUTTON */
    div.stButton > button {
        height: 60px;
        background-color: #1e2127;
        border: 2px solid #5DADE2;
        border-radius: 12px;
        color: #A0AAB5;
        font-size: 34px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        border-color: #00FF00;
        color: #00FF00;
        background-color: #1e2127;
    }

    /* DISABLE LIGHT MODE TOGGLE */
    div[role="dialog"] [data-testid="stWidgetLabel"] + div[role="radiogroup"] {
        display: none !important;
    }
    
    /* MISC CLEANUP */
    hr {
        border-top: 1px solid #30363d !important;
        margin-top: 0rem !important;
        margin-bottom: 1rem !important;
    }

    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# --- 10. HEADER ---
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
current_dir = Path(__file__).parent
image_path = current_dir / "assets" / "Team_logo.png"
img_base64 = get_image_base64(image_path) if image_path.exists() else ""
col_title, col_status = st.columns([4, 1.4])

with col_title:
    st.markdown(
        '<h1 style="margin-bottom: 0;"><i class="bi bi-activity" style="color: #5DADE2; margin-right: 12px;"></i>GDP Nowcast Terminal</h1>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div style='color: #a1a1aa; font-size: 15px; margin-top: 5px;'>"
        f"DSE3101 &nbsp;|&nbsp; "
        f"<i class='bi bi-arrow-repeat' style='margin-right: 5px;'></i>"
        f"<b>Last Sync:</b> {datetime.now().strftime('%H:%M:%S')}"
        f"</div>",
        unsafe_allow_html=True
    )

with col_status:
    btn_col, logo_col = st.columns([1, 1])

    with logo_col:
        if img_base64:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; align-items: center; margin-top: 15px;">
                    <img src="data:image/png;base64,{img_base64}" width="100">
                </div>
                """,
                unsafe_allow_html=True
            )

    with btn_col:
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

        refresh_clicked = st.button(
            "⟳ Refresh data",
            use_container_width=True,
            key="refresh_btn_header"
        )
st.divider()

# --- 11. SIDEBAR ---
with st.sidebar:
    page = option_menu(
        menu_title=None,  # Hides the title to keep it clean like the image
        options=["Live Statistics", "Monthly Nowcast", "History Chart"],
        icons=["graph-up-arrow", "calendar4", "bar-chart-line"], 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"font-size": "16px"}, 
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "margin": "4px 0", 
                "border-radius": "8px", # Gives that pill-shaped highlight
                "--hover-color": "rgba(255, 255, 255, 0.05)"
            },
            "nav-link-selected": {
                "background-color": "#4b5563", # Matches the dark grey highlight in your image
                "font-weight": "bold"
            },
        }
    )
    
    st.divider()
    st.markdown(
    '<h3><i class="bi bi-gear" style="margin-right: 8px; color: #a1a1aa;"></i>Configuration</h3>', 
    unsafe_allow_html=True
)
    
    # Initialize variables to None BEFORE checking the page
    hist_params = None
    selected_quarter = None
    
    # Conditional Sidebar Logic
    if page == "History Chart":
        hist_params = history_chart.get_sidebar_controls(gdp_data)
        config_panel.render() 
        
    elif page == "Live Statistics":
        st.markdown("<p style='color:#a1a1aa; font-size: 14px; margin-bottom: 5px;'>Prediction Intervals</p>", unsafe_allow_html=True)
        show_50 = st.toggle("50% Interval", value=False)
        show_80 = st.toggle("80% Interval", value=False)
        
    elif page == "Monthly Nowcast":
        selected_quarter = intra_quarter_chart.get_sidebar_filters()

# --- TRIGGER DIALOG BEFORE CONTENT ---
if st.session_state.get("success_popup", False):
    st.session_state["success_popup"] = False  # Reset immediately
    success_popup()

# --- 12. MAIN CONTENT AREA ---
from frontend.components.fred_nowcast import get_fred_data, render_fred_card
from frontend.components.live_metric import (
    load_live_nowcast_df,
    get_latest_bridge_value,
)

# --- GLOBAL REFRESH LOGIC (Moved outside the page tabs!) ---
if refresh_clicked:
    st.toast("Starting data pipeline...", icon="🚀")
    with st.spinner("Accessing FRED API & Re-running Models..."):
        try:
            api_script = ROOT_DIR / "src" / "api_preprocessing.py"
            model_script = ROOT_DIR / "src" / "live_nowcast.py"
            evo_script = ROOT_DIR / "frontend" / "export_bridge_evolution.py" 
            
            #subprocess.run([sys.executable, str(api_script)], check=True)
            #subprocess.run([sys.executable, str(model_script)], check=True)
            #subprocess.run([sys.executable, str(evo_script)], check=True) 

            env = os.environ.copy()
            env["FRED_API_KEY"] = st.secrets["FRED_API_KEY"]

            subprocess.run([sys.executable, str(api_script)], check=True, env=env)
            subprocess.run([sys.executable, str(model_script)], check=True, env=env)
            subprocess.run([sys.executable, str(evo_script)], check=True, env=env)
            
            st.session_state["success_popup"] = True
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# --- PAGE ROUTING ---
if page == "Live Statistics":
    live_df = load_live_nowcast_df()
    quarter, bridge_val = get_latest_bridge_value(live_df)
    atl_val, atl_quarter, stl_val, stl_quarter = get_fred_data()

    col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1, 1, 1, 1, 1])

    with col1:
        biz_cycle.render(gdp_data)
    with col2:
        live_metric.render_bridge_card()
    with col3:
        live_metric.render_ar_card()
    with col4:
        live_metric.render_adl_card()
    with col5:
        render_fred_card("Atlanta GDPNow", atl_val, quarter)
    with col6:
        render_fred_card("St. Louis Fed", stl_val, quarter)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    live_graph.render(show_50, show_80)

elif page == "Monthly Nowcast":
    st.markdown("<br>", unsafe_allow_html=True)
    intra_quarter_chart.render(gdp_data, selected_quarter)

elif page == "History Chart":
    st.markdown("<br>", unsafe_allow_html=True)
    
    if hist_params:
        history_chart.render(gdp_data, *hist_params)
    else:
        st.error("Sidebar controls failed to load.")