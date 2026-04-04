import streamlit as st
import pandas as pd
import numpy as np
from frontend.components.fred_industry_models import get_historical_nowcasts

def get_fred_data(target_quarter):
    """
    Fetches historical data and filters for the specific quarter 
    provided by the bridge model.
    """
    df_historical = get_historical_nowcasts()
    
    if df_historical.empty or target_quarter not in df_historical.index:
        return None, None

    try:
        # Pull specific scalars for the target quarter
        gdp_now = df_historical.loc[target_quarter, "Atlanta Fed Forecast"]
        st_louis = df_historical.loc[target_quarter, "St. Louis Fed Forecast"]
        
        return float(gdp_now), float(st_louis)
    except (KeyError, ValueError, TypeError):
        return None, None
def render_fred_card(label, value, quarter):
    """
    Renders a standalone metric card that matches the 
    styling of the live_metric card.
    """
    # Unit correction (0.02 -> 2.00)
    display_val = value
    if display_val is not None and abs(display_val) < 0.5:
        display_val *= 100

    # Formatting the string
    val_text = f"{display_val:.2f}%" if display_val is not None else "N/A"

    # Display using the custom-card CSS already defined in main.py
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
            {label} ({quarter if quarter else 'N/A'})
        </div>
        <div style="color: white; font-size: 28px; font-weight: bold;">
            {val_text}
        </div>
    </div>
    """, unsafe_allow_html=True)