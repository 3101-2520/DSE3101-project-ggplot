import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from .atlanta_fed import get_historical_nowcasts

# --- 1. NEW UNIVERSAL CSV LOADER (Added back from yesterday) ---
@st.cache_data
def load_model_csv(filename):
    """Loads a pre-computed prediction CSV from the data folder."""
    try:
        csv_path = Path(__file__).resolve().parents[2] / "data" / filename
        df = pd.read_csv(csv_path)
        
        # Dynamically find the columns
        q_col = [c for c in df.columns if 'quarter' in c.lower() or 'date' in c.lower()][0]
        p_col = [c for c in df.columns if 'predict' in c.lower() or 'forecast' in c.lower()][0]
        
        # Format dates properly so they don't crash the chart
        clean_quarters = df[q_col].astype(str).str.replace(" ", "")
        df[q_col] = pd.PeriodIndex(clean_quarters, freq='Q').astype(str).str.replace("Q", " Q")
        
        df.set_index(q_col, inplace=True)
        return df[p_col]
        
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return pd.Series()

def render(gdp_growth):
    nowcasts_df = get_historical_nowcasts()

    current_q_period = pd.Timestamp.now().to_period("Q")
    current_q_label = str(current_q_period).replace("Q", " Q")
    
    # Team's new logic for cleaning up the current quarter
    if not nowcasts_df.empty and "Real GDP (Actual)" in nowcasts_df.columns:
        nowcasts_df = nowcasts_df.copy()
        if current_q_label in nowcasts_df.index:
            nowcasts_df.at[current_q_label, "Real GDP (Actual)"] = None
            
    if "Atlanta Fed Forecast" in nowcasts_df.columns:
        prev_q_label = str(current_q_period - 1).replace("Q", " Q")
        if prev_q_label in nowcasts_df.index:
            q1_val = nowcasts_df.at[current_q_label, "Atlanta Fed Forecast"]
            q4_val = nowcasts_df.at[prev_q_label, "Atlanta Fed Forecast"]
            if q1_val == q4_val:
                pass

    st.markdown("### Chart Controls")
    col1, col2, col3 = st.columns(3)

    min_year = int(gdp_growth.index.min().year)
    max_year = int(gdp_growth.index.max().year)

    with col1:
        year = st.number_input("Select Year", min_value=min_year, max_value=max_year, value=2024, step=1)

    with col2:
        q = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])

    with col3:
        window_size = st.number_input("Quarters to display", min_value=3, max_value=21, value=7, step=2)

    selected_period = pd.Period(f"{year}{q}", freq="Q")

    if selected_period not in gdp_growth.index:
        st.warning(f"Selected period {year} {q} is not in your dataset yet.")
        return

    half_window = window_size // 2
    full_periods = pd.period_range(start=selected_period - half_window, end=selected_period + half_window, freq="Q")
    full_labels = [str(p).replace("Q", " Q") for p in full_periods]

    if not nowcasts_df.empty:
        valid_nowcasts = nowcasts_df.reindex(full_labels)
    else:
        valid_nowcasts = pd.DataFrame(index=full_labels)

    fig = go.Figure()
    active_models = st.session_state.get("active_models", [])

    # --- TEAM'S NEW ACTUAL GDP LOGIC ---
    if "Real GDP (Actual)" in valid_nowcasts.columns:
        fig.add_trace(go.Scatter(
            x=valid_nowcasts.index,
            y=valid_nowcasts["Real GDP (Actual)"],
            mode="lines+markers",
            name="US Real GDP (FRED)",
            line=dict(color="#5DADE2", width=4), # Changed to a nice solid blue
            marker=dict(size=8, symbol="circle"),
            connectgaps=False 
        ))

    # --- AR MODEL (Using Fast CSV Loader) ---
    if "AR Model" in active_models:
<<<<<<< HEAD
        ar_preds = load_model_csv("historical_gdp_ar_predictions.csv")
        if not ar_preds.empty:
            ar_zoom = (ar_preds * 100).reindex(full_labels) # Scaled to %
            fig.add_trace(go.Scatter(
                x=full_labels,         
                y=ar_zoom.values,      
                mode="lines+markers",
                name="AR Model (Benchmark)",
                line=dict(dash="dash", width=2, color="#E74C3C"), 
                marker=dict(size=6),
                connectgaps=False
            ))
=======
        ar_preds = get_ar_predictions(gdp_growth)
        ar_scaled = ar_preds 
        ar_zoom = ar_scaled.reindex(full_periods)
>>>>>>> shannon


    # --- ADL MODEL (Using Fast CSV Loader) ---
    if "ADL Model" in active_models:
        adl_preds = load_model_csv("historical_gdp_adl_predictions.csv")
        if not adl_preds.empty:
            adl_zoom = (adl_preds * 100).reindex(full_labels) # Scaled to %
            fig.add_trace(go.Scatter(
                x=full_labels,         
                y=adl_zoom.values,      
                mode="lines+markers",
                name="ADL Model",
                line=dict(dash="dashdot", width=2.5, color="#9B59B6"), 
                marker=dict(size=7, symbol="square"),
                connectgaps=False
            ))

    # --- BRIDGE MODEL (Using Fast CSV Loader) ---
    if "Bridge Model" in active_models:
        bridge_preds = load_model_csv("historical_gdp_bridge_predictions.csv")
        if not bridge_preds.empty:
            bridge_zoom = (bridge_preds * 100).reindex(full_labels) # Scaled to %
            fig.add_trace(go.Scatter(
                x=full_labels,         
                y=bridge_zoom.values,      
                mode="lines+markers",
                name="Bridge Model",
                line=dict(dash="longdash", width=2.5, color="#F1C40F"), 
                marker=dict(size=7, symbol="diamond"),
                connectgaps=False
            ))

    # --- FED FORECASTS (With / 4 Scaling) ---
    if "Atlanta Fed" in active_models and "Atlanta Fed Forecast" in valid_nowcasts.columns:
        fig.add_trace(go.Scatter(
            x=valid_nowcasts.index,
            y=valid_nowcasts["Atlanta Fed Forecast"] / 4, # Scaled down
            mode="lines+markers",
            name="Atlanta Fed (GDPNow)",
            line=dict(dash="dot", width=3, color="#E67E22"),
            marker=dict(size=8),
            connectgaps=False
        ))

    if "St. Louis Fed" in active_models and "St. Louis Fed Forecast" in valid_nowcasts.columns:
        fig.add_trace(go.Scatter(
            x=valid_nowcasts.index,
            y=valid_nowcasts["St. Louis Fed Forecast"] / 4, # Scaled down
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
        yaxis_title="Growth (%)",
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