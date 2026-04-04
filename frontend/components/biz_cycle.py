import streamlit as st
import pandas as pd
from pathlib import Path


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
    text_color = "#FFFFFF"
    flash_class = "" # Default: no flashing

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
                    flash_class = "flash-red"
                elif current < 0:
                    label = "Contracting"
                    text_color = "#FF3333" # Neon Red
                    flash_class = "flash-red"
                elif current >= 0 and current >= v1:
                    label = "Expansion"
                    text_color = "#00FF00" # Neon Green
                    flash_class = "flash-green"
                elif current >= 0 and current < v1:
                    label = "Decelerating Growth"
                    text_color = "#F1C40F" # Yellow
                    flash_class = "flash-yellow"
        except Exception:
            pass 

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

    # 4. Display the card
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
        <div style="color: #A0AAB5; font-size: 14px; text-transform: uppercase; margin-bottom: 8px;">
            Live Business Cycle ({quarter_str if quarter_str else 'N/A'})
        </div>
        <div class="{flash_class}" style="color: {text_color}; font-size: 28px; font-weight: bold;">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)