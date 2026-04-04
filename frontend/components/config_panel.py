import streamlit as st
from utils import apply_custom_font

# This single function call applies IBM Plex Mono to the entire file
apply_custom_font()

def render():
    st.subheader("Model Configuration")
    st.markdown("Select the nowcasting models to display:")

    # Create the tickboxes (These will all automatically use IBM Plex Mono)
    with st.container():
        ar_model = st.checkbox("AR Model (Benchmark)", value=True)
        adl_model = st.checkbox("ADL Model", value=False)      
        bridge_model = st.checkbox("Bridge Model", value=True) 
        atl_fed = st.checkbox("Atlanta Fed (GDPNow)", value=False)
        stl_fed = st.checkbox("St. Louis Fed Forecast", value=False)

    # Compile the list
    current_selection = []
    if ar_model: current_selection.append("AR Model")
    if adl_model: current_selection.append("ADL Model")
    if bridge_model: current_selection.append("Bridge Model")
    if atl_fed: current_selection.append("Atlanta Fed")
    if stl_fed: current_selection.append("St. Louis Fed")

    st.session_state['active_models'] = current_selection

    st.divider()
    
    if len(current_selection) == 0:
        st.warning("⚠️ No models selected")
    else:
        st.caption(f"Tracking **{len(current_selection)}** active models.")