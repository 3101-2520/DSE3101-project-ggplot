import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .atlanta_fed import get_historical_nowcasts

# --- 1. IMPORT YOUR AR MODEL ---
from models.ar_benchmark import fit_ar_benchmark

# --- 2. CACHE THE PREDICTIONS ---
# We cache this so the math only runs once per session
@st.cache_data
def get_ar_predictions(gdp_series):
    # Train the model on the full historical dataset
    model = fit_ar_benchmark(gdp_series)
    # Return the historical predictions (fitted values)
    return model.fittedvalues

def render(gdp_growth):
    nowcasts_df = get_historical_nowcasts()

    st.markdown("### Chart Controls")
    col1, col2, col3 = st.columns(3)

    min_year = int(gdp_growth.index.min().year)
    max_year = int(gdp_growth.index.max().year)

    with col1:
        year = st.number_input(
            "Select Year",
            min_value=min_year,
            max_value=max_year,
            value=2024,
            step=1
        )

    with col2:
        q = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])

    with col3:
        window_size = st.number_input(
            "Quarters to display",
            min_value=3,
            max_value=21,
            value=7,
            step=2
        )

    selected_period = pd.Period(f"{year}{q}", freq="Q")

    if selected_period not in gdp_growth.index:
        st.warning(f"Selected period {year} {q} is not in your dataset yet.")
        return

    half_window = window_size // 2

    full_periods = pd.period_range(
        start=selected_period - half_window,
        end=selected_period + half_window,
        freq="Q"
    )

    full_labels = [str(p).replace("Q", " Q") for p in full_periods]

    hist_zoom = (gdp_growth * 100).reindex(full_periods).to_frame(name="Growth")
    hist_zoom["label"] = full_labels

    if not nowcasts_df.empty:
        valid_nowcasts = nowcasts_df.reindex(full_labels)
    else:
        valid_nowcasts = pd.DataFrame(index=full_labels)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_zoom["label"],
        y=hist_zoom["Growth"],
        mode="lines+markers",
        name="Actual GDP",
        line=dict(color="#5DADE2", width=4),
        marker=dict(size=8),
        connectgaps=False
    ))

    active_models = st.session_state.get("active_models", [])

    if "AR Model" in active_models:
        # 1. Get predictions and scale to percentages
        ar_preds = get_ar_predictions(gdp_growth)
        ar_scaled = ar_preds * 100
        
        # 2. Reindex using your new windowing logic!
        ar_zoom = ar_scaled.reindex(full_periods)

        # 3. Draw the line
        fig.add_trace(go.Scatter(
            x=full_labels,         # Using your new full_labels list
            y=ar_zoom.values,      # Using the reindexed values
            mode="lines+markers",
            name="AR Model (Benchmark)",
            line=dict(dash="dash", width=2, color="#E74C3C"), # Sharp red
            marker=dict(size=6),
            connectgaps=False
        ))
    # ---------------------------------------------
    if "Atlanta Fed" in active_models and "Atlanta Fed Forecast" in valid_nowcasts.columns:
        fig.add_trace(go.Scatter(
            x=valid_nowcasts.index,
            y=valid_nowcasts["Atlanta Fed Forecast"],
            mode="lines+markers",
            name="Atlanta Fed (GDPNow)",
            line=dict(dash="dot", width=3, color="#E67E22"),
            marker=dict(size=8),
            connectgaps=False
        ))

    if "St. Louis Fed" in active_models and "St. Louis Fed Forecast" in valid_nowcasts.columns:
        fig.add_trace(go.Scatter(
            x=valid_nowcasts.index,
            y=valid_nowcasts["St. Louis Fed Forecast"],
            mode="lines+markers",
            name="St. Louis Fed Forecast",
            line=dict(dash="dot", width=3, color="#58D68D"),
            marker=dict(size=8),
            connectgaps=False
        ))

    fig.update_layout(
        title=f"GDP Growth & Forecasts around {year} {q}",
        template="plotly_dark",
        hovermode="x unified",
        xaxis_title="Quarter",
        yaxis_title="GDP Growth (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    fig.update_xaxes(categoryorder="array", categoryarray=full_labels)

    st.plotly_chart(fig, use_container_width=True)