import streamlit as st
import pandas as pd
from fredapi import Fred

@st.cache_data(ttl=3600)
def get_fred_data(quarter):
    """
    Fetches the official Nowcasts directly from the LIVE FRED API.
    """
    try:
        # Grabs your API key directly from .streamlit/secrets.toml
        fred = Fred(api_key=st.secrets["FRED_API_KEY"])
        
        # Ping the API for both series
        atl_series = fred.get_series('GDPNOW')
        stl_series = fred.get_series('STLENI')
        
        # Drop any empty dates and grab the absolute latest prediction
        atl_val = float(atl_series.dropna().iloc[-1]) if not atl_series.dropna().empty else None
        stl_val = float(stl_series.dropna().iloc[-1]) if not stl_series.dropna().empty else None
        
        return atl_val, stl_val
        
    except Exception as e:
        print(f"Error connecting to FRED: {e}")
        return None, None
    
def render_fred_card(label, value, quarter):
    # Formats the raw API value safely (multiplier removed to prevent the 40% bug!)
    val_text = f"{value:.2f}%" if value is not None else "N/A"

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