import pandas as pd
import streamlit as st
from fredapi import Fred

@st.cache_data(ttl=3600)
def get_fed_nowcast_data():
    try:
        api_key = st.secrets["FRED_API_KEY"]
        fred = Fred(api_key=api_key)
        
        series_map = {
            'Atlanta Fed (GDPNow)': 'GDPNOW',
            'St. Louis Fed (Nowcast)': 'STLENI'
        }
        
        df = pd.DataFrame()
        for label, s_id in series_map.items():
            df[label] = fred.get_series(s_id).tail(12) # Fetch recent months
        return df.ffill()
    except Exception as e:
        st.error(f"Fed Data Error: {e}")
        return pd.DataFrame()