import streamlit as st
from config import *

def get_fred_client():
    """
    Return a FRED API client using the API key stored in the environment variable FRED_API_KEY.

    To set the key:
        - On Windows (PowerShell): $env:FRED_API_KEY = "your_key_here"
        - On Linux/macOS: export FRED_API_KEY="your_key_here"
    """
    api_key = st.secrets.get("FRED_API_KEY") or os.environ.get("FRED_API_KEY")
    if api_key is None:
        raise ValueError("FRED_API_KEY environment variable not set")
    return Fred(api_key=api_key)

def fetch_monthly_series(series_ids, start_date, end_date, sleep_seconds=0.5):
    """
    Fetch monthly series from FRED one by one.
    Returns:
        df: DataFrame with DatetimeIndex
        failed_ids: list of series IDs that could not be fetched
    """
    fred = get_fred_client()
    series_list = []
    failed_ids = []

    for i, sid in enumerate(series_ids):
        print(f"Fetching {sid} ({i+1}/{len(series_ids)})...")
        try:
            data = fred.get_series(
                sid,
                observation_start=start_date,
                observation_end=end_date
            )
            data = pd.to_numeric(data, errors="coerce")
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()

            # Convert everything to monthly frequency.
            # For higher-frequency series (daily/weekly), keep the last observation in each month.
            # Label each month at month start to match your MD CSV style.
            data = data.resample("MS").last()

            # Re-apply requested window after resampling
            data = data.loc[start_date:end_date]

            data.name = sid
            series_list.append(data)
        except Exception as e:
            print(f"  Failed to fetch {sid}: {e}")
            failed_ids.append(sid)

        time.sleep(sleep_seconds)

    if series_list:
        df = pd.concat(series_list, axis=1)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    else:
        df = pd.DataFrame()

    return df, failed_ids

def fetch_quarterly_gdp(series_id='GDPC1', start_date='1960-01-01', end_date=None):
    """
    Fetch quarterly GDP series.
    Returns a pandas Series with DatetimeIndex.
    """
    # Dynamically set end_date to today if not explicitly provided
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    fred = get_fred_client()
    gdp = fred.get_series(
        series_id,
        observation_start=start_date,
        observation_end=end_date
    )
    gdp = pd.to_numeric(gdp, errors="coerce")
    gdp.index = pd.to_datetime(gdp.index)
    gdp = gdp.sort_index()
    return gdp