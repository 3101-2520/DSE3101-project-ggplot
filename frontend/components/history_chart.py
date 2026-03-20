from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.data_preprocessing import load_and_transform_qd

# Load and transform historical data to get quarterly GDP growth rates
GDP_growth = load_and_transform_qd(str(ROOT_DIR / "data" / "2026-02-QD.csv"))

st.title("Quarterly GDP Growth")

# Inputs
col1, col2 = st.columns(2)

period_index = GDP_growth.index
min_year = int(period_index.min().year)
max_year = int(period_index.max().year)

with col1:
    selected_year = st.number_input(
        "Year",
        min_value=min_year,
        max_value=max_year,
        value=2020,
        step=1
    )

with col2:
    selected_quarter = st.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])

selected_period = pd.Period(f"{selected_year}{selected_quarter}", freq="Q")
selected_label = str(selected_period).replace("Q", " Q")

if selected_period in period_index:
    selected_loc = period_index.get_loc(selected_period)

    # 7 quarters total = selected quarter ± 3 quarters
    window_size = 7
    half_window = 3

    start_loc = selected_loc - half_window
    end_loc = selected_loc + half_window

    # If too close to the start, shift right
    if start_loc < 0:
        end_loc += -start_loc
        start_loc = 0

    # If too close to the end, shift left
    if end_loc > len(period_index) - 1:
        shift_left = end_loc - (len(period_index) - 1)
        start_loc -= shift_left
        end_loc = len(period_index) - 1

    # Final safeguard
    start_loc = max(0, start_loc)

    zoom_series = GDP_growth.iloc[start_loc:end_loc + 1]
    zoom_df = (zoom_series * 100).to_frame(name="GDP_growth").reset_index()

    # Label x-axis as e.g. 2020 Q4
    zoom_df["quarter_label"] = (
        zoom_df["sasdate"].astype(str).str.replace("Q", " Q", regex=False)
    )

    zoom_fig = px.line(
        zoom_df,
        x="quarter_label",
        y="GDP_growth",
        title=f"GDP Growth around {selected_label}",
        labels={"GDP_growth": "GDP Growth (%)", "quarter_label": "Quarter"},
        hover_data={"GDP_growth": ":.2f"}
    )

    zoom_fig.update_traces(
        mode="lines+markers",
        hovertemplate="<b>%{x}</b><br>GDP Growth: %{y:.2f}%<extra></extra>"
    )

    zoom_fig.add_hline(y=0, line_dash="dash")

    zoom_fig.update_yaxes(
        range=[-10.5, 10.5],
        tickmode="array",
        tickvals=[-10, -5, 0, 5, 10],
        ticksuffix="%",
        showgrid=True
    )

    st.plotly_chart(zoom_fig, use_container_width=True)
else:
    st.warning("Selected quarter is not in the dataset.")

# To run, paste in terminal: streamlit run frontend/components/history_chart.py