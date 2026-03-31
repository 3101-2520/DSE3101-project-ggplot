import streamlit as st
from datetime import date
import pandas as pd


def render(bridge_history_df):
  year = st.session_state["use selected year"]
  q = st.session_state["use selected q"]
 
  selected_period = pd.Period(f"{year}{q}", freq="Q")

  # Get bridge
  bridge_history_df["period"] = pd.PeriodIndex(
    bridge_history_df["Year and Quarter"].str.replace(" ", ""),
    freq="Q"
  )

  row = bridge_history_df[
    bridge_history_df["period"] == selected_period
  ]

  # Get previous 2 quarters
  p1 = selected_period - 1
  p2 = selected_period - 2

  # Extract rows
  row1 = bridge_history_df[bridge_history_df["period"] == p1]
  row2 = bridge_history_df[bridge_history_df["period"] == p2]

  if not row1.empty and not row2.empty and not row.empty:
    current = row["Bridge predicted GDP growth"].iloc[0]
    v1 = row1["Bridge predicted GDP growth"].iloc[0]
    v2 = row2["Bridge predicted GDP growth"].iloc[0]

    #logic
    if v2 < v1 < current:
        label = "growth"
        bg_color = "#A8E6A3"
        text_color = "black"

    elif v2 > v1 > current:
        label = "recession"
        bg_color = "#F5A3A3"
        text_color = "black"

    else:
        label = "-"
        bg_color = "#333333"
        text_color = "white"

  else:
        label = "-"
        bg_color = "#333333"
        text_color = "white"

  # Display metric
  st.markdown(f"""
  <div style="
      background-color: {bg_color};
      padding: 20px;
      border-radius: 12px;
      text-align: center;
  ">
      <div style="color: {text_color}; font-size: 16px;">
          Business Cycle
      </div>
      <div style="color: {text_color}; font-size: 32px; font-weight: bold;">
          {label}
      </div>
  </div>
  """, unsafe_allow_html=True)
