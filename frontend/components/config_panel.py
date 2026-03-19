import streamlit as st

def render():
    with st.expander("MODEL CONFIGURATION", expanded=True):
        model_type = st.selectbox("Select Model", ["AR Model", "Dynamic Factor", "Bridge Equation"])
        
        descriptions = {
            "AR Model": "Autoregressive Model - Uses historical GDP patterns to forecast future values based on past lags.",
            "Dynamic Factor": "Extracts common trends from a large set of economic indicators.",
            "Bridge Equation": "Links high-frequency indicators directly to quarterly GDP growth."
        }
        
        st.info(descriptions[model_type])