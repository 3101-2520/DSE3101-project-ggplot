import streamlit as st
import pandas as pd
from pathlib import Path

# --- 1. DATA LOADER ---
@st.cache_data
def get_live_value():
    """Fetches the latest prediction from the bridge model flashes."""
    try:
        # Resolves path to data/live_nowcast_results.csv
        csv_path = Path(__file__).resolve().parents[2] / "data" / "live_nowcast_results.csv"
        df = pd.read_csv(csv_path)
        if df.empty: 
            return None, None
        
        row = df.iloc[0] # Grab the latest quarter row
        
        # Check flashes in reverse order (Flash 3 -> 2 -> 1) to get the most recent data
        if pd.notna(row.get('bridge_flash3')): 
            return row['quarter'], row['bridge_flash3']
        if pd.notna(row.get('bridge_flash2')): 
            return row['quarter'], row['bridge_flash2']
        if pd.notna(row.get('bridge_flash1')): 
            return row['quarter'], row['bridge_flash1']
        
        return row['quarter'], None
    except Exception:
        return None, None

# --- 2. MAIN RENDER ---
def render():
    """Renders the stylized metric card for the main dashboard."""
    quarter, value = get_live_value()
    
    # Scale correction: If the backend provides decimals (0.02 instead of 2.0), 
    # we multiply by 100 for a consistent percentage display.
    if value is not None and abs(value) < 0.5:
        value = value * 100

    # Display the metric card using HTML/CSS
    st.markdown(f"""
    <div style="
        background-color: #1e2127;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #30363d;
        height: 114px; /* Matches the height of your Refresh Button */
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="color: #A0AAB5; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">
            Current Bridge Nowcast ({quarter if quarter else 'N/A'})
        </div>
        <div style="color: #00ff00; font-size: 32px; font-weight: bold; margin-top: 5px;">
            {f"{value:.2f}%" if value is not None else "Awaiting..."}
        </div>
    </div>
    """, unsafe_allow_html=True)