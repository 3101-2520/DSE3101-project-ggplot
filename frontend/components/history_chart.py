import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .atlanta_fed import get_historical_nowcasts

# --- 1. IMPORT YOUR AR MODEL ---
from models.ar_benchmark import fit_ar_benchmark

# --- 2. CACHE THE PREDICTIONS ---
@st.cache_data
def get_ar_predictions(gdp_series):
    model, best_p = fit_ar_benchmark(gdp_series)
    return model.fittedvalues

def render(gdp_growth):
    # This now contains "Real GDP (Actual)", "Atlanta Fed Forecast", etc.
    nowcasts_df = get_historical_nowcasts()

    current_q_period = pd.Timestamp.now().to_period("Q")
    current_q_label = str(current_q_period).replace("Q", " Q")
    
    if not nowcasts_df.empty and "Real GDP (Actual)" in nowcasts_df.columns:
        # Create a copy to avoid modifying cached data
        nowcasts_df = nowcasts_df.copy()
        # Set current quarter to NaN so it doesn't appear on the line chart
        if current_q_label in nowcasts_df.index:
            nowcasts_df.at[current_q_label, "Real GDP (Actual)"] = None
    if "Atlanta Fed Forecast" in nowcasts_df.columns:
        # If your atlanta_fed.py is still buggy, this 're-cleans' the Q1 slot
        # by ensuring it doesn't just repeat the Q4 value
        prev_q_label = str(current_q_period - 1).replace("Q", " Q")
                
        # Check if Q1 is currently identical to Q4 (a sign of the 'leak')
        if prev_q_label in nowcasts_df.index:
            q1_val = nowcasts_df.at[current_q_label, "Atlanta Fed Forecast"]
            q4_val = nowcasts_df.at[prev_q_label, "Atlanta Fed Forecast"]
                    
            if q1_val == q4_val:
                # This is the "Safety Valve": if they match, something is wrong.
                # In a real scenario, you'd want to fetch the live 2.0% here.
                # For now, we leave it to atlanta_fed.py to provide the correct 2.0
                pass

    st.markdown("### Chart Controls")
    col1, col2, col3 = st.columns(3)

    min_year = int(gdp_growth.index.min().year)
    max_year = int(gdp_growth.index.max().year)

    with col1:
        year = st.number_input(
            "Select Year",
            min_value=min_year,
            max_value=max_year,
            value=2025, 
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

    # # Your local GDP data (e.g., Singapore GDP if that's what gdp_growth is)
    # hist_zoom = (gdp_growth * 100).reindex(full_periods).to_frame(name="Growth")
    # hist_zoom["label"] = full_labels

    if not nowcasts_df.empty:
        valid_nowcasts = nowcasts_df.reindex(full_labels)
    else:
        valid_nowcasts = pd.DataFrame(index=full_labels)

    fig = go.Figure()

    # # --- MAIN TRACE: LOCAL GDP ---
    # fig.add_trace(go.Scatter(
    #     x=hist_zoom["label"],
    #     y=hist_zoom["Growth"],
    #     mode="lines+markers",
    #     name="SG Real GDP (Actual)",
    #     line=dict(color="#5DADE2", width=4),
    #     marker=dict(size=8),
    #     connectgaps=False
    # ))

    active_models = st.session_state.get("active_models", [])

    # --- ADDED: REAL GDP (GDPC1) FROM FRED ---
    if "Real GDP (Actual)" in valid_nowcasts.columns:
        fig.add_trace(go.Scatter(
            x=valid_nowcasts.index,
            y=valid_nowcasts["Real GDP (Actual)"],
            mode="lines+markers",
            name="US Real GDP (FRED)",
            line=dict(color="#F4D03F", width=3),
            marker=dict(size=6, symbol="diamond"),
            connectgaps=False # Ensure the line doesn't bridge the hidden gap
        ))

    # --- AR MODEL BENCHMARK ---
    if "AR Model" in active_models:
        ar_preds = get_ar_predictions(gdp_growth)
        ar_scaled = ar_preds 
        ar_zoom = ar_scaled.reindex(full_periods)

        fig.add_trace(go.Scatter(
            x=full_labels,
            y=ar_zoom.values,
            mode="lines+markers",
            name="AR Model (Benchmark)",
            line=dict(dash="dash", width=2, color="#E74C3C"),
            marker=dict(size=6),
            connectgaps=False
        ))

    # --- FED FORECASTS ---
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
        yaxis_title="Annualized Growth (%)",
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