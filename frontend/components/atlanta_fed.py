import pandas as pd
import streamlit as st
from fredapi import Fred

@st.cache_data(ttl=3600)
def get_historical_nowcasts():
    """Fetches historical nowcasts and resamples them to one point per quarter."""
    try:
        api_key = st.secrets["FRED_API_KEY"]
        fred = Fred(api_key=api_key)
        
        series_map = {
            'Atlanta Fed Forecast': 'GDPNOW',
            'St. Louis Fed Forecast': 'STLENI'
        }
        
        combined_df = pd.DataFrame()
        
        for label, series_id in series_map.items():
            try:
                # Grab ~3 years of history to ensure it covers your chart window
                series_data = fred.get_series(series_id)
                combined_df[label] = series_data.tail(1000) 
            except Exception:
                continue
                
        if combined_df.empty:
            return combined_df

        # 1. Sort chronologically
        combined_df = combined_df.sort_index()

        # 2. CRITICAL FIX: Resample to Quarterly (Takes the final estimate for that quarter)
        combined_df = combined_df.resample('Q').last()

        # 3. Format index to exactly match your CSV labels (e.g., "2024 Q1")
        combined_df.index = (
            combined_df.index.to_period('Q')
            .astype(str)
            .str.replace("Q", " Q")
        )
        
        return combined_df.dropna(how='all')
        
    except Exception as e:
        st.error(f"Error fetching Fed data: {e}")
        return pd.DataFrame()