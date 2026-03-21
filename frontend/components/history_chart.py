import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .atlanta_fed import get_historical_nowcasts

def render(gdp_growth):
    # 1. Fetch the resampled historical nowcasts
    nowcasts_df = get_historical_nowcasts()
    
    # 2. Get Selection from config_panel
    year = st.session_state.get('selected_year', 2024)
    q = st.session_state.get('selected_q', 'Q1')
    selected_period = pd.Period(f"{year}{q}", freq="Q")

    if selected_period not in gdp_growth.index:
        st.warning("Selected period not in dataset.")
        return

    # 3. Filter Historical Window
    idx = gdp_growth.index.get_loc(selected_period)
    start, end = max(0, idx-3), min(len(gdp_growth)-1, idx+3)
    
    hist_zoom = (gdp_growth.iloc[start:end+1] * 100).to_frame(name="Growth")
    hist_zoom["label"] = hist_zoom.index.astype(str).str.replace("Q", " Q")

    # Define the "Next Quarter" label for the upcoming, unreleased forecast
    last_actual_period = hist_zoom.index[-1]
    next_period = last_actual_period + 1
    next_label = str(next_period).replace("Q", " Q")

    # Define the full X-axis range we want to show on the chart
    all_x_labels = hist_zoom["label"].tolist() + [next_label]

    # Filter Fed data so the chart doesn't zoom out to show 3 years of Fed data
    # when you only want to see a 7-quarter window.
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

    # TRACES: Historical Fed Nowcasts (Dashed lines tracking the same timeline)
    if not valid_nowcasts.empty:
        for col in valid_nowcasts.columns:
            col_data = valid_nowcasts[col].dropna() # Drop gaps
            fig.add_trace(go.Scatter(
                x=col_data.index, 
                y=col_data.values,
                mode="lines+markers",
                name=col,
                line=dict(dash="dot", width=2),
                marker=dict(size=6)
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
# To run, paste in terminal: streamlit run frontend/components/history_chart.py