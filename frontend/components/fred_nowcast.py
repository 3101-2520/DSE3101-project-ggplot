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
    display_val = value
    if display_val is not None and abs(display_val) < 0.5:
        display_val *= 100

    val_text = f"{display_val:.2f}%" if display_val is not None else "N/A"

    st.markdown(f"""
    <div style="
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
        <div style="
            color: #A0AAB5;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
            min-height: 54px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        ">
            {label} ({quarter if quarter else 'N/A'})
        </div>
        <div style="
            color: white;
            font-size: 28px;
            font-weight: bold;
            line-height: 1;
        ">
            {val_text}
        </div>
    </div>
    """, unsafe_allow_html=True)