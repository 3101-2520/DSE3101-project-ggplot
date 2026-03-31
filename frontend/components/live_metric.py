import streamlit as st
from datetime import date
import pandas as pd


def render(bridge_history_df):
  year = st.session_state["selected year"]
  q = st.session_state["selected q"]
 
  selected_period = pd.Period(f"{year}{q}", freq="Q")

  # Get bridge
  bridge_history_df["period"] = pd.PeriodIndex(
    bridge_history_df["Year and Quarter"].str.replace(" ", ""),
    freq="Q"
  )

  row = bridge_history_df[
    bridge_history_df["period"] == selected_period
  ]

  if not row.empty:
    value = row["Bridge predicted GDP growth"].iloc[0]
  else:
    value = None
 
  # Display metric
  
  st.markdown(f"""
  <div style="
      background-color: #EDEDED;
      padding: 20px;
      border-radius: 12px;
      text-align: center;
  ">
      <div style="color: black; font-size: 16px;">
          Bridge nowcast of {year} {q}
      </div>
      <div style="color: black; font-size: 32px; font-weight: bold;">
          {round(value, 2) if value is not None else "-"}
      </div>
  </div>
  """, unsafe_allow_html=True)
