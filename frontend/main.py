import streamlit as st
from components import config_panel, live_metric, history_chart
from datetime import datetime

st.set_page_config(layout="wide", page_title="GDP Nowcast Terminal")

# Custom CSS for that clean "Terminal" look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# Top Header
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("GDP Nowcast Terminal")
    st.markdown("GG(PLOT) | <span style='background-color:#90EE90; padding:2px 8px; border-radius:10px;'>⚡ MARKET BOOM</span>", unsafe_allow_html=True)

with col_status:
    st.write(f"Last Updated: {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}")

st.divider()

# Main Layout Grid
top_left, top_right = st.columns([1, 2])

with top_left:
    config_panel.render()

with top_right:
    live_metric.render()

st.container()
history_chart.render()