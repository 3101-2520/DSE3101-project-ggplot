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

def render_model_card(title, quarter, value, height=120, tooltip_text=""):
    
    # Updated color and flashing logic
    if value is not None and pd.notna(value):
        if value < 0:
            text_color = "red"
            flash_class = "flash-red" if "Bridge" in title else ""
        else:
            text_color = "#00A86B"
            flash_class = "flash-green" if "Bridge" in title else ""
    else:
        text_color = "#A0AAB5"
        flash_class = ""

    st.markdown("""
    <style>
    /* Pulsating Animations */
    @keyframes pulse-green {
        0% { color: #00A86B; text-shadow: 0 0 0px rgba(0, 168, 107, 0); }
        50% { color: #33ffaa; text-shadow: 0 0 15px rgba(0, 168, 107, 0.6); }
        100% { color: #00A86B; text-shadow: 0 0 0px rgba(0, 168, 107, 0); }
    }

    @keyframes pulse-red {
        0% { color: red; text-shadow: 0 0 0px rgba(255, 0, 0, 0); }
        50% { color: #ff6666; text-shadow: 0 0 15px rgba(255, 0, 0, 0.6); }
        100% { color: red; text-shadow: 0 0 0px rgba(255, 0, 0, 0); }
    }

    .flash-green {
        animation: pulse-green 2s infinite ease-in-out;
    }

    .flash-red {
        animation: pulse-red 2s infinite ease-in-out;
    }

    .card-container {
        position: relative;
        background-color: #1e2127;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #30363d;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
        cursor: help;
    }

    /* Tooltip text container */
    .card-container .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #30363d;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 100;
        bottom: 105%; 
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
        font-weight: 300;
        border: 1px solid #A0AAB5;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
    }

    /* Show tooltip on hover */
    .card-container:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card-container" style="height: {height}px;">
        <span class="tooltiptext">{tooltip_text}</span>
        <div style="
            color: #A0AAB5;
            font-size: 11px;
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
    render_model_card("Current Bridge Nowcast", quarter, value, tooltip_text="Bridge Model: Acts as a 'bridge' between high-frequency indicators and low-frequency indicators", height=120)

def render_ar_card():
    df = load_live_nowcast_df()
    ref_quarter = get_reference_quarter(df)
    quarter, value = get_column_value_for_quarter(df, ref_quarter, "ar_benchmark")
    render_model_card("Current AR Nowcast", quarter, value, tooltip_text="Autoregressive Model: Benchmark model that predicts based on past indicators", height=120)

def render_adl_card():
    df = load_live_nowcast_df()
    ref_quarter = get_reference_quarter(df)
    quarter, value = get_column_value_for_quarter(df, ref_quarter, "adl_benchmark")
    render_model_card("Current ADL Nowcast", quarter, value, tooltip_text="Autoregressive Distributed Lag Model: Extension of AR Model to include past values of other explanatory variables", height=120)