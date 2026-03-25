import pandas as pd
import streamlit as st
import requests
import certifi


def fetch_fred_series(series_id: str, api_key: str) -> pd.Series:
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

    # Keep only date + value
    df = df[["date", "value"]].copy()

    # Convert types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date", "value"]).set_index("date")

    return df["value"]


@st.cache_data(ttl=3600)
def get_historical_nowcasts():
    """Fetch historical nowcasts and resample to one point per quarter."""
    try:
        api_key = st.secrets["FRED_API_KEY"]

        series_map = {
            "Atlanta Fed Forecast": "GDPNOW",
            "St. Louis Fed Forecast": "STLENI",
        }

        combined_df = pd.DataFrame()

        for label, series_id in series_map.items():
            try:
                series_data = fetch_fred_series(series_id, api_key)
                combined_df[label] = series_data.tail(1000)
            except Exception as e:
                st.error(f"🚨 Failed to fetch {label}: {e}")
                continue

        if combined_df.empty:
            return combined_df

        combined_df = combined_df.sort_index()
        combined_df = combined_df.resample("Q").last()

        combined_df.index = (
            combined_df.index.to_period("Q")
            .astype(str)
            .str.replace("Q", " Q", regex=False)
        )

        return combined_df.dropna(how="all")

    except Exception as e:
        st.error(f"Error fetching Fed data: {e}")
        return pd.DataFrame()