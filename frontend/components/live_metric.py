# import streamlit as st
# from datetime import date
# import pandas as pd

# def render(bridge_history_df):
#   year = st.session_state["selected year"]
#   q = st.session_state["selected q"]
 
#   selected_period = pd.Period(f"{year}{q}", freq="Q")

#   # Get bridge
#   bridge_history_df["period"] = pd.PeriodIndex(
#     bridge_history_df["Year and Quarter"].str.replace(" ", ""),
#     freq="Q"
#   )

#   row = bridge_history_df[
#     bridge_history_df["period"] == selected_period
#   ]

#   if not row.empty:
#     value = row["Bridge predicted GDP growth"].iloc[0]
#   else:
#     value = None
 
#   # Display metric
#   st.markdown(f"""
#   <div style="
#       background-color: #EDEDED;
#       padding: 20px;
#       border-radius: 12px;
#       text-align: center;
#   ">
#       <div style="color: black; font-size: 16px;">
#           Bridge nowcast of {year} {q}
#       </div>
#       <div style="color: black; font-size: 32px; font-weight: bold;">
#           {round(value, 2) if value is not None else "-"}
#       </div>
#   </div>
#   """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

def render():
    csv_path = ROOT_DIR / "data" / "historical_gdp_ar_predictions.csv"

    if not csv_path.exists():
        st.markdown("AR metric unavailable")
        return

    df = pd.read_csv(csv_path)

    if df.empty or "AR benchmark predicted GDP growth" not in df.columns:
        st.markdown("AR metric unavailable")
        return

    df = df.dropna(subset=["AR benchmark predicted GDP growth"])

    if df.empty:
        st.markdown("AR metric unavailable")
        return

    latest_row = df.iloc[-1]
    quarter = latest_row["Year and Quarter"]
    value = latest_row["AR benchmark predicted GDP growth"]

    st.markdown(f"""
    <div style="
        background-color: #EDEDED;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    ">
        <div style="color: black; font-size: 16px;">
            Latest AR benchmark prediction ({quarter})
        </div>
        <div style="color: black; font-size: 32px; font-weight: bold;">
            {round(value, 2)}
        </div>
    </div>
    """, unsafe_allow_html=True)
