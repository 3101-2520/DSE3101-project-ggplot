import streamlit as st
from datetime import datetime
import sys
from pathlib import Path
import requests
import certifi

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="GDP Nowcast Terminal")

# --- 2. PATH FIX ---
# Ensure we can see 'src' and 'frontend' from the root
ROOT_DIR = Path(__file__).resolve().parents[1] 
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# --- 3. DATA LOADING (Centralized) ---
from src.data_preprocessing import load_and_transform_qd

@st.cache_data
def get_historical_data():
    csv_path = ROOT_DIR / "data" / "2026-02-QD.csv"
    return load_and_transform_qd(str(csv_path))

gdp_data = get_historical_data()

# --- 4. COMPONENT IMPORTS ---
try:
    from frontend.components import config_panel, live_metric, history_chart
except ModuleNotFoundError:
    from components import config_panel, live_metric, history_chart

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

#with top_right:
    # Renders the small metric cards at the top
    #live_metric.render()

st.container()
# Pass gdp_data to the main chart for the integrated view
history_chart.render(gdp_data)