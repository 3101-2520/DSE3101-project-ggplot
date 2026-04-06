import streamlit as st
import pandas as pd
from fredapi import Fred
import re
import requests
import certifi

    
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
        # St. Louis: FRED is fine
        fred = Fred(api_key=st.secrets["FRED_API_KEY"])
        stl_series = fred.get_series("STLENI")

        stl_clean = stl_series.dropna()
        if not stl_clean.empty:
            stl_val = float(stl_clean.iloc[-1])
            stl_quarter = stl_clean.index[-1].to_period("Q").strftime("%Y Q%q")

    except Exception as e:
        print(f"Error fetching St. Louis nowcast: {e}")

    return atl_val, atl_quarter, stl_val, stl_quarter
    
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