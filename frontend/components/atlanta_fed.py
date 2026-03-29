import pandas as pd
import streamlit as st
import requests
import certifi


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
            # -----------------------------------

            # Resample to Quarter End (QE)
            s_q = s.resample("QE").last()

            # Handle the 'Now' (2026 Q1) data
            latest_val = s.iloc[-1]
            current_q_end = pd.Timestamp.now().to_period("Q").to_timestamp("Q")
            s_q[current_q_end] = latest_val

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

        # Return the last 20 quarters
        return combined_df

    except Exception as e:
        st.error(f"Error fetching Fed data: {e}")
        return pd.DataFrame()