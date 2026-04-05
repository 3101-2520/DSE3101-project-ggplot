import streamlit as st
import pandas as pd
from pathlib import Path

@st.cache_data
def load_live_nowcast_df():
    try:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "live_nowcast_results.csv"
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def scale_if_needed(value):
    if value is not None and pd.notna(value) and abs(value) < 0.5:
        return value * 100
    return value


def get_latest_bridge_value(df):
    """Get latest available bridge nowcast using flash3 -> flash2 -> flash1."""
    if df is None or df.empty:
        return None, None

    for _, row in df.iloc[::-1].iterrows():
        quarter = row.get("quarter", None)
        if pd.notna(row.get("bridge_flash3")):
            return quarter, row["bridge_flash3"]
        if pd.notna(row.get("bridge_flash2")):
            return quarter, row["bridge_flash2"]
        if pd.notna(row.get("bridge_flash1")):
            return quarter, row["bridge_flash1"]

    return None, None


def get_row_for_quarter(df, quarter):
    """Return the row matching a given quarter."""
    if df is None or df.empty or quarter is None:
        return None

    matched = df[df["quarter"].astype(str) == str(quarter)]
    if matched.empty:
        return None
    return matched.iloc[0]


def get_bridge_value_for_quarter(df, quarter):
    """Get bridge value for a specific quarter."""
    row = get_row_for_quarter(df, quarter)
    if row is None:
        return quarter, None

    if pd.notna(row.get("bridge_flash3")):
        return quarter, row["bridge_flash3"]
    if pd.notna(row.get("bridge_flash2")):
        return quarter, row["bridge_flash2"]
    if pd.notna(row.get("bridge_flash1")):
        return quarter, row["bridge_flash1"]

    return quarter, None


def get_column_value_for_quarter(df, quarter, column_name):
    """Get AR or ADL value for the same selected quarter only."""
    row = get_row_for_quarter(df, quarter)
    if row is None or column_name not in row.index:
        return quarter, None

    value = row.get(column_name, None)
    if pd.notna(value):
        return quarter, value
    return quarter, None


def render_model_card(title, quarter, value, height=120):
    value = scale_if_needed(value)

    text_color = "#00ff00"
    flash_class = "flash-green"

    if value is not None and pd.notna(value):
        if value < 0:
            text_color = "#FF3333"
            flash_class = "flash-red"
    else:
        text_color = "#A0AAB5"
        flash_class = ""

    st.markdown(f"""
    <div style="
        background-color: #1e2127;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #30363d;
        height: {height}px;
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
            {title} ({quarter if quarter else 'N/A'})
        </div>
        <div class="{flash_class}" style="
            color: {text_color};
            font-size: 28px;
            font-weight: bold;
            line-height: 1;
        ">
            {f"{value:.2f}%" if value is not None and pd.notna(value) else "Awaiting..."}
        </div>
    </div>
    """, unsafe_allow_html=True)


def get_reference_quarter(df):
    """
    Use the latest available bridge quarter as the common quarter
    for all model cards.
    """
    quarter, _ = get_latest_bridge_value(df)
    return quarter


def render_bridge_card():
    df = load_live_nowcast_df()
    ref_quarter = get_reference_quarter(df)
    quarter, value = get_bridge_value_for_quarter(df, ref_quarter)
    render_model_card("Current Bridge Nowcast", quarter, value, height=120)


def render_ar_card():
    df = load_live_nowcast_df()
    ref_quarter = get_reference_quarter(df)
    quarter, value = get_column_value_for_quarter(df, ref_quarter, "ar_benchmark")
    render_model_card("Current AR Nowcast", quarter, value, height=120)


def render_adl_card():
    df = load_live_nowcast_df()
    ref_quarter = get_reference_quarter(df)
    quarter, value = get_column_value_for_quarter(df, ref_quarter, "adl_benchmark")
    render_model_card("Current ADL Nowcast", quarter, value, height=120)