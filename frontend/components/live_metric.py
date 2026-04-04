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

    # Determine Color and Flash Class based on value
    text_color = "#00ff00"  # Default Green
    flash_class = "flash-green"
    
    if value is not None:
        if value < 0:
            text_color = "#FF3333" # Neon Red
            flash_class = "flash-red"
    else:
        text_color = "#A0AAB5"
        flash_class = ""

    # 3. Inject CSS Animations (Matching the Biz Cycle file)
    st.markdown("""
    <style>
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
    .flash-green {
        animation: pulse-green 2s infinite;
    }
    .flash-red {
        animation: pulse-red 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)

    # 4. Display the metric card
    st.markdown(f"""
    <div style="
        background-color: #1e2127;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #30363d;
        height: 100px; 
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="color: #A0AAB5; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
            Current Bridge Nowcast ({quarter if quarter else 'N/A'})
        </div>
        <div class="{flash_class}" style="color: {text_color}; font-size: 28px; font-weight: bold;">
            {f"{value:.2f}%" if value is not None else "Awaiting..."}
        </div>
    </div>
    """, unsafe_allow_html=True)