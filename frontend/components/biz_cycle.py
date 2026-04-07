import streamlit as st
import pandas as pd
from pathlib import Path

def get_all_cycle_descriptions():
    """Returns succinct definitions for all business cycle phases."""
    return (
        "<b>Recession:</b> 2+ quarters of negative growth.<br>"
        "<b>Contracting:</b> Current growth is negative.<br>"
        "<b>Expansion:</b> Growth is positive & accelerating.<br>"
        "<b>Slowing:</b> Growth is positive but slowing."
    )

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
        label=""
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
                    label = "Decelerating"
                    text_color = "#F1C40F" # Yellow
                    flash_class = "flash-yellow"
        except Exception:
            pass 

    # 3. Get the static tooltip text containing all definitions
    tooltip_text = get_all_cycle_descriptions()
    
    # 4. Display the card with UNIQUE class names
    st.markdown("""
    <style>
    .cycle-card-container {
        position: relative;
        cursor: help;
    }
    .cycle-card-container .cycle-tooltiptext {
        visibility: hidden;
        width: 350px !important; 
        min-width: 350px !important;
        white-space: normal !important;
        background-color: #30363d;
        color: #fff;
        text-align: center; 
        border-radius: 8px;
        padding: 16px;
        position: absolute;
        z-index: 9999 !important; 
        bottom: 110%; 
        left: 50%; 
        margin-left: -175px; 
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 13px;
        font-weight: 400;
        border: 1px solid #A0AAB5;
        line-height: 1.5;
    }
    .cycle-card-container:hover .cycle-tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    # 5. Render the Card using the UNIQUE class names
    st.markdown(f"""
    <div class="cycle-card-container" style="
        background-color: #1e2127;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #30363d;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
    ">
        <span class="cycle-tooltiptext">{tooltip_text}</span>
        <div style="
            color: #A0AAB5;
            font-size: 14px;
            font-weight: bold; 
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
            min-height: 54px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        ">
            Live Business Cycle ({quarter_str if quarter_str else 'N/A'})
        </div>
        <div class="{flash_class}" style="
            color: {text_color};
            font-size: 24px; 
            font-weight: bold;
            line-height: 1;
        ">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)