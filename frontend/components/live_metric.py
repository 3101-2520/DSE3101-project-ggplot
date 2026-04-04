import streamlit as st
import pandas as pd
from pathlib import Path
from utils import apply_custom_font

apply_custom_font()

@st.cache_data
def get_live_value():
    try:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "live_nowcast_results.csv"
        df = pd.read_csv(csv_path)
        if df.empty: return None, None
        
        row = df.iloc[0] # Current quarter
        
        # Grab the latest available flash
        if pd.notna(row.get('bridge_flash3')): return row['quarter'], row['bridge_flash3']
        if pd.notna(row.get('bridge_flash2')): return row['quarter'], row['bridge_flash2']
        if pd.notna(row.get('bridge_flash1')): return row['quarter'], row['bridge_flash1']
        
        return row['quarter'], None
    except Exception:
        return None, None

def render():
    quarter, value = get_live_value()
    
    # Smart multiplier just in case backend hasn't fixed the decimals yet
    if value is not None and abs(value) < 0.5:
        value = value * 100

    st.markdown(f"""
    <div style="
        background-color: #1e2127;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #30363d;
        font-family: 'IBM Plex Mono', monospace; /* <--- EXPLICITLY SET THE FONT HERE */
    ">
        <div style="color: #A0AAB5; font-size: 16px;">
            Current Bridge Nowcast ({quarter if quarter else 'N/A'})
        </div>
        <div style="color: #00ff00; font-size: 32px; font-weight: bold;">
            {f"{value:.2f}%" if value is not None else "N/A"}
        </div>
    </div>
    """, unsafe_allow_html=True)