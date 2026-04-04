import pandas as pd
import streamlit as st
import requests
import certifi
from utils import apply_custom_font

apply_custom_font()

def fetch_fred_series(series_id: str, api_key: str) -> pd.Series:
    """Fetches the standard, clean historical series from FRED."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    
    response = requests.get(url, params=params, verify=certifi.where(), timeout=30)
    response.raise_for_status()
    data = response.json()
    
    observations = data.get("observations", [])
    if not observations:
        return pd.Series(dtype="float64")

    df = pd.DataFrame(observations)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    
    # We drop NAs and set the index to the date (usually start of quarter for these)
    return df.dropna(subset=["value"]).set_index("date")["value"]

@st.cache_data(ttl=3600)
def get_historical_nowcasts() -> pd.DataFrame:
    try:
        api_key = st.secrets["FRED_API_KEY"]
        
        # Added GDPC1 for Real GDP
        series_map = {
            "Real GDP (Actual)": "GDPC1",
            "Atlanta Fed Forecast": "GDPNOW",
            "St. Louis Fed Forecast": "STLENI",
        }

        resampled_dict = {}

        for label, series_id in series_map.items():
            s = fetch_fred_series(series_id, api_key)
            if s.empty: continue

            # --- CUSTOM LOGIC FOR ACTUAL GDP ---
            if series_id == "GDPC1":
                # Convert absolute levels to Quarter-over-Quarter Annualized Rate
                # This makes it comparable to the Fed forecasts (e.g., 2.5%)
                s = s.pct_change()
                s = ((1 + s)**4 - 1) * 100 
            # 2. Get the ACTUAL latest value (the 2.0%)
            live_val = s.iloc[-1] # This is the 2.0

            # Create the resampled frame
            s_q = s.resample("QE").last()

            # FORCE the current quarter to be the live_val
            # This is the most important line to stop the 4.2 leak!
            current_q_end = pd.Timestamp.now().to_period("Q").to_timestamp("Q") + pd.offsets.QuarterEnd(0)
            s_q[current_q_end] = live_val

            resampled_dict[label] = s_q

        if not resampled_dict:
            return pd.DataFrame()

        combined_df = pd.DataFrame(resampled_dict).sort_index()

        # Format index to "2025 Q4"
        combined_df.index = (
            combined_df.index.to_period("Q")
            .astype(str)
            .str.replace("Q", " Q", regex=False)
        )

        return combined_df

    except Exception as e:
        st.error(f"Error fetching Fed data: {e}")
        return pd.DataFrame()
    

def annualize_gdp_growth(gdp_level):
    gdp_growth = ((gdp_level / gdp_level.shift(1)) ** 4 - 1) * 100
    gdp_growth.name = "GDP_growth"
    return gdp_growth
