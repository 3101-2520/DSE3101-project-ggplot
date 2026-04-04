import streamlit as st

def render():
    st.markdown("Nowcast Models:")

    # Create the tickboxes 
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