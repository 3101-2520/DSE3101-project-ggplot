import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .atlanta_fed import get_historical_nowcasts

def render(gdp_growth):
    # 1. Fetch the resampled historical nowcasts
    nowcasts_df = get_historical_nowcasts()
    
    # --- RESTORED DATE SELECTION UI ---
    st.markdown("### Chart Controls")
    col1, col2 = st.columns(2)
    
    # Dynamically get the available years from your CSV
    min_year = int(gdp_growth.index.min().year)
    max_year = int(gdp_growth.index.max().year)

    with col1:
        year = st.number_input("Select Year", min_value=min_year, max_value=max_year, value=2024, step=1)
    with col2:
        q = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])
        
    selected_period = pd.Period(f"{year}{q}", freq="Q")

    if selected_period not in gdp_growth.index:
        st.warning(f"Selected period {year} {q} is not in your dataset yet.")
        return

    # 3. Filter Historical Window
    idx = gdp_growth.index.get_loc(selected_period)
    start, end = max(0, idx-3), min(len(gdp_growth)-1, idx+3)
    
    hist_zoom = (gdp_growth.iloc[start:end+1] * 100).to_frame(name="Growth")
    hist_zoom["label"] = hist_zoom.index.astype(str).str.replace("Q", " Q")

    # Lock the X-axis strictly to the 7 periods of actual GDP data
    all_x_labels = hist_zoom["label"].tolist()

    # Filter Fed data so the chart doesn't zoom out
    if not nowcasts_df.empty:
        valid_nowcasts = nowcasts_df[nowcasts_df.index.isin(all_x_labels)]
    else:
        valid_nowcasts = pd.DataFrame()

    # 4. Create the Unified Figure
    fig = go.Figure()

    # TRACE: Actual GDP (Solid blue line)
    fig.add_trace(go.Scatter(
        x=hist_zoom["label"],
        y=hist_zoom["Growth"],
        mode="lines+markers",
        name="Actual GDP",
        line=dict(color="#5DADE2", width=4),
        marker=dict(size=8)
    ))

    # --- LINK TICKBOXES TO THE GRAPH ---
    # Safely get the list of checked boxes from config_panel.py
    active_models = st.session_state.get('active_models', [])

    # TRACE: Atlanta Fed 
    if not valid_nowcasts.empty:
        # ONLY draw it if the user ticked the box AND the data exists
        if "Atlanta Fed" in active_models and 'Atlanta Fed Forecast' in valid_nowcasts.columns:
            col_data = valid_nowcasts['Atlanta Fed Forecast'].dropna()
            fig.add_trace(go.Scatter(
                x=col_data.index, 
                y=col_data.values,
                mode="lines+markers",
                name="Atlanta Fed (GDPNow)",
                line=dict(dash="dot", width=3, color="#E67E22"), # Orange to stand out
                marker=dict(size=8)
            ))

    # 5. Dashboard Styling
    fig.update_layout(
        title=f"GDP Growth & Forecasts around {year} {q}",
        template="plotly_dark", 
        hovermode="x unified",
        xaxis_title="Quarter",
        yaxis_title="GDP Growth (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    fig.update_xaxes(categoryorder="array", categoryarray=all_x_labels)

    st.plotly_chart(fig, use_container_width=True)