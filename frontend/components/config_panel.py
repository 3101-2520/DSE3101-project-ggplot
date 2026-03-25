import streamlit as st

def render():
    st.subheader("Model Configuration")
    st.markdown("Select the nowcasting models to display:")

    # Create the tickboxes (Checkboxes)
    # We use containers to keep the layout neat inside your top_left column
    with st.container():
        ar_model = st.checkbox("AR Model (Benchmark)", value=True)
        atl_fed = st.checkbox("Atlanta Fed (GDPNow)", value=True)
        stl_fed = st.checkbox("St. Louis Fed Forecast", value=False)
        hist_mean = st.checkbox("Historical Mean", value=False)

    # Compile the list of what is currently checked
    current_selection = []
    if ar_model: current_selection.append("AR Model")
    if atl_fed: current_selection.append("Atlanta Fed")
    if stl_fed: current_selection.append("St. Louis Fed")
    if hist_mean: current_selection.append("Historical Mean")

    # --- THE CRITICAL PART ---
    # Save this list to Streamlit's "global memory" (Session State)
    # This allows your live_metric and history_chart files to read these choices
    st.session_state['active_models'] = current_selection

    st.divider()
    
    # A small status indicator to match your terminal aesthetic
    if len(current_selection) == 0:
        st.warning("⚠️ No models selected")
    else:
        st.caption(f"Tracking **{len(current_selection)}** active models.")