import streamlit as st
import pandas as pd
from fredapi import Fred
import re
import requests
import certifi
import os

@st.cache_data(ttl=3600)
def get_fred_data():
    """
    Fetch live nowcasts:
    - Atlanta from the official Atlanta Fed GDPNow page
    - St. Louis from FRED
    """
    atl_val = None
    stl_val = None
    atl_quarter = None
    stl_quarter = None

    try:
        # Atlanta: official source
        atl_url = "https://www.atlantafed.org/research-and-data/data/gdpnow"
        atl_resp = requests.get(atl_url, verify=certifi.where(), timeout=30)
        atl_resp.raise_for_status()
        atl_text = atl_resp.text

        quarter_match = re.search(r"Latest GDPNow Estimate for (\d{4}:Q[1-4])", atl_text)
        value_match = re.search(r'(\d+(?:\.\d+)?)%</span>\s*<div[^>]*>\s*Latest GDPNow Estimate', atl_text)

        if quarter_match:
            atl_quarter = quarter_match.group(1).replace(":", " ")

        if value_match:
            atl_val = float(value_match.group(1))
        else:
            # fallback if page structure shifts slightly
            text_only = re.sub(r"<[^>]+>", " ", atl_text)
            generic_value = re.search(r"(\d+(?:\.\d+)?)%\s*Latest GDPNow Estimate", text_only)
            if generic_value:
                atl_val = float(generic_value.group(1))

    except Exception as e:
        print(f"Error fetching Atlanta GDPNow: {e}")

    try:
        # St. Louis: FRED
        api_key = st.secrets.get("FRED_API_KEY") or os.environ.get("FRED_API_KEY")
        if not api_key:
            raise ValueError("FRED_API_KEY not found in Streamlit secrets or environment variables.")
        fred = Fred(api_key=api_key)
        stl_series = fred.get_series("STLENI")

        stl_clean = stl_series.dropna()
        if not stl_clean.empty:
            stl_val = float(stl_clean.iloc[-1])
            stl_quarter = stl_clean.index[-1].to_period("Q").strftime("%Y Q%q")

    except Exception as e:
        print(f"Error fetching St. Louis nowcast: {e}")

    return atl_val, atl_quarter, stl_val, stl_quarter

def get_fred_description(label):
    """Returns definitions for external Fed nowcasting models."""
    descriptions = {
        "Atlanta GDPNow": "A running estimate of real GDP growth based on available economic data for the current measured quarter.",
        "St. Louis Fed": "The St. Louis Fed's Economic News Index (STLENI), which tracks real-time economic conditions."
    }
    return descriptions.get(label, "External benchmark for GDP Nowcasting.")
    
def render_fred_card(label, value, quarter):
    # Formats the raw API value safely (multiplier removed to prevent the 40% bug!)
    val_text = f"{value:.2f}%" if value is not None else "N/A"
    tooltip_text = get_fred_description(label)
    
    # Determine the color based on the value
    if value is None:
        val_color = "#A0AAB5"  # Default grey for missing data
    elif value < 0:
        val_color = "red"      # Negative
    else:
        val_color = "#00A86B"  # Positive or zero

    st.markdown("""
    <style>
    .card-container {
        position: relative;
        cursor: help;
    }
    .card-container .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #30363d;
        color: #fff;
        text-align: center;
        border-radius: 8px;
        padding: 10px;
        position: absolute;
        z-index: 100;
        bottom: 110%; 
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 13px;
        font-weight: 300; 
        border: 1px solid #A0AAB5;
        line-height: 1.4;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
    }
    .card-container:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    # 3. Render Card
    st.markdown(f"""
    <div class="card-container" style="
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
        <span class="tooltiptext">{tooltip_text}</span>
        <div style="
            color: #A0AAB5;
            font-size: 12px;
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
            {label} ({quarter if quarter else 'N/A'})
        </div>
        <div style="
            color: {val_color}; /* Dynamically set color based on value */
            font-size: 28px;
            font-weight: bold; 
            line-height: 1;
        ">
            {val_text}
        </div>
    </div>
    """, unsafe_allow_html=True)