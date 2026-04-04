import streamlit as st
import pandas as pd
from utils import apply_custom_font
from pathlib import Path

apply_custom_font()

def render(gdp_data):
    # 1. Fetch the Live Prediction directly from the CSV
    try:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "live_nowcast_results.csv"
        live_df = pd.read_csv(csv_path)
        if live_df.empty:
            raise ValueError("No live data")
            
        row = live_df.iloc[0]
        quarter_str = str(row['quarter']).strip()
        
        mult = 100 if abs(row.get('ar_benchmark', 0)) < 0.5 else 1
        
        # Get the latest available flash
        current = None
        for flash in ['bridge_flash3', 'bridge_flash2', 'bridge_flash1']:
            if pd.notna(row.get(flash)):
                current = row[flash] * mult
                break
                
        if current is None:
            raise ValueError("No flash predictions available")
            
    except Exception:
        current = None
        quarter_str = None

    # 2. Execute the Logic using PROPER Macroeconomic Rules
    label = "-"
    text_color = "white"

    if current is not None and quarter_str is not None:
        try:
            live_period = pd.Period(quarter_str, freq="Q")
            p1 = live_period - 1 # The previous quarter

            # Look up ACTUAL GDP growth from the historical series
            if p1 in gdp_data.index:
                v1 = gdp_data.loc[p1]

                # Real-world business cycle logic
                if current < 0 and v1 < 0:
                    label = "Recession"
                    text_color = "#FF3333" # Neon Red
                elif current < 0:
                    label = "Contracting"
                    text_color = "#FF3333" # Neon Red
                elif current >= 0 and current >= v1:
                    label = "Expansion"
                    text_color = "#00FF00" # Neon Green
                elif current >= 0 and current < v1:
                    label = "Decelerating Growth"
                    text_color = "#F1C40F" # Yellow
        except Exception:
            pass 

    # 3. Display the metric card
    st.markdown(f"""
    <div style="
        background-color: #1e2127;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #30363d;
        font-family: 'IBM Plex Mono', monospace; 
    ">
        <div style="color: #A0AAB5; font-size: 16px;">
            Live Business Cycle ({quarter_str if quarter_str else 'N/A'})
        </div>
        <div style="color: {text_color}; font-size: 32px; font-weight: bold;">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)