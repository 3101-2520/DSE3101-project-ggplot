import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import *
from src.FRED_API_pipeline import fetch_monthly_series, fetch_quarterly_gdp

def csv_to_api(name):
    if name.endswith("x"):
        name=name[:-1]
    name = name.replace(' ', '_')
    mapping = {
            'S&P_500': 'SP500',
            'S&P_div_yield': 'SP500DY',
            'S&P_PE_ratio': 'SP500PE'
            }
    name = mapping.get(name, name)
    # unavailable series
    unavailable = {"HWI", "HWIURATIO", "CONSPI", "CLAIMS", "AMDMNO", "COMPAPFF"}
    if name in unavailable:
        return None
    return name

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    md_path = ROOT / "data/2026-02-MD.csv"
    fetch_start = "1960-01-01"
    fetch_end = datetime.today().strftime("%Y-%m-%d")
    print(f"Fetching monthly series from {fetch_start} to {fetch_end}...")

    # Load CSV
    md_meta = pd.read_csv(md_path, nrows=1)
    csv_cols = [c for c in md_meta.columns if c != "sasdate"]
    print("CSV variables:", len(csv_cols))

    # Map to API names

    csv_to_api_map = {c: csv_to_api(c) for c in csv_cols}
    api_ids = [v for v in csv_to_api_map.values() if v is not None]
    print("API series to fetch:", len(api_ids))

    # Fetch from API

    raw_monthly, failed = fetch_monthly_series(api_ids, fetch_start, fetch_end, sleep_seconds = 0.1)
    print("Fetched series:", len(raw_monthly.columns))
    last_available = raw_monthly.index.max()
    print(f"Last available date in existing data: {last_available}")
    latest_values = raw_monthly.loc[last_available].notna().sum()
    print(f"Number of series with data at last available date: {latest_values}")


    if failed: 
        print("Failed API series:", failed)

    # Rename API IDs to CSV names

    api_to_csv = {v: k for k, v in csv_to_api_map.items() if v is not None}
    raw_monthly = raw_monthly.rename(columns=api_to_csv)
    print("Columns after rename:", len(raw_monthly.columns))

    # Save monthly dataset
    out_path = ROOT / "data/live_api_monthly.csv"
    raw_monthly = raw_monthly.sort_index()
    raw_monthly.to_csv(out_path)
    print(f"Saved API monthly data to {out_path}")
    print("Shape:", raw_monthly.shape)

    # Fetch quarterly GDP
    gdp_out = ROOT / "data/live_api_quarterly_gdp.csv"
    gdp_raw = fetch_quarterly_gdp(series_id='GDPC1',start_date=fetch_start, end_date=fetch_end)
    gdp_raw.to_csv(gdp_out, header = True)

    print(f"Saved API quarterly GDP to {gdp_out}")
    print("Shape:", gdp_raw.shape)

    last_gdp_date = gdp_raw.index.max() if 'gdp_raw' in locals() and not gdp_raw.empty else 'N/A'
    print(f"Last available date in GDP data: {last_gdp_date}")