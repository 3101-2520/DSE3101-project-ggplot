import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from .fred_industry_models import get_historical_nowcasts

@st.cache_data
def load_model_csv(filename):
    """Loads a pre-computed prediction CSV from the data folder."""
    try:
        csv_path = Path(__file__).resolve().parents[2] / "data" / filename
        df = pd.read_csv(csv_path)
        
        # Dynamically find the columns
        q_col = [c for c in df.columns if 'quarter' in c.lower() or 'date' in c.lower()][0]
        p_col = [c for c in df.columns if 'predict' in c.lower() or 'forecast' in c.lower()][0]
        
        # Format dates properly
        clean_quarters = df[q_col].astype(str).str.replace(" ", "")
        df[q_col] = pd.PeriodIndex(clean_quarters, freq='Q').astype(str).str.replace("Q", " Q")
        
        df.set_index(q_col, inplace=True)
        return df[p_col]
    except Exception:
        return pd.Series(dtype=float)
    
@st.cache_data
def load_live_nowcast_csv():
    """Loads live_nowcast_results.csv and formats the quarter index."""
    try:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "live_nowcast_results.csv"
        df = pd.read_csv(csv_path)

        clean_quarters = df["quarter"].astype(str).str.replace(" ", "", regex=False)
        df["quarter"] = pd.PeriodIndex(clean_quarters, freq="Q").astype(str).str.replace("Q", " Q")
        df.set_index("quarter", inplace=True)

        return df
    except Exception:
        return pd.DataFrame()
    
def overwrite_last_row_with_live(history_series, live_df, live_col):
    """
    Replace only the final row of the history series with the corresponding
    value from live_nowcast_results.csv, if that quarter exists there.
    """
    if history_series.empty or live_df.empty or live_col not in live_df.columns:
        return history_series

    aligned = history_series.copy()
    last_hist_q = aligned.index[-1]

    if last_hist_q in live_df.index and pd.notna(live_df.loc[last_hist_q, live_col]):
        aligned.loc[last_hist_q] = live_df.loc[last_hist_q, live_col] * 100

    return aligned
    
def get_prediction_max_period():
    """Return the latest quarter available across AR, ADL, and Bridge prediction CSVs."""
    series_list = [
        load_model_csv("historical_gdp_ar_predictions.csv"),
        load_model_csv("historical_gdp_adl_predictions.csv"),
        load_model_csv("historical_gdp_bridge_predictions.csv"),
    ]

    max_periods = []
    for s in series_list:
        if not s.empty:
            clean_idx = s.index.astype(str).str.replace(" ", "", regex=False)
            periods = pd.PeriodIndex(clean_idx, freq="Q")
            max_periods.append(periods.max())

    if max_periods:
        return max(max_periods)

    return None

def get_sidebar_controls(gdp_data):
    """Call this inside st.sidebar to centralize controls"""
    prediction_max_period = get_prediction_max_period()

    if prediction_max_period is None:
        prediction_max_period = gdp_data.index.max()

    default_year = prediction_max_period.year
    default_q = f"Q{prediction_max_period.quarter}"

    min_year = int(gdp_data.index.min().year)
    max_year = int(prediction_max_period.year)

    col1, col2 = st.columns(2)

    with col1:
        year_options = list(range(min_year, max_year + 1))
        default_year_index = year_options.index(default_year) if default_year in year_options else len(year_options) - 1

        year = st.selectbox(
            "Year",
            options=year_options,
            index=default_year_index
        )

    q_options = ["Q1", "Q2", "Q3", "Q4"]

    if year == prediction_max_period.year:
        q_options = [f"Q{i}" for i in range(1, prediction_max_period.quarter + 1)]

    default_q_index = q_options.index(default_q) if default_q in q_options else len(q_options) - 1

    with col2:
        q = st.selectbox(
            "Quarter",
            options=q_options,
            index=default_q_index
        )

    window_options = list(range(3, 23, 2))
    default_window_index = window_options.index(7) if 7 in window_options else 2

    window = st.selectbox(
        "Display Qtrs",
        options=window_options,
        index=default_window_index
    )

    return year, q, window

def render(gdp_data, year, q, window_size):
    """Main render function receiving parameters from sidebar"""
    nowcasts_df = get_historical_nowcasts()
    live_nowcasts_df = load_live_nowcast_csv()
    
    prediction_max_period = get_prediction_max_period()
    if prediction_max_period is None:
        prediction_max_period = gdp_data.index.max()
    
    # Calculate time window
    try:
        selected_period = pd.Period(f"{year}{q}", freq="Q")
    except ValueError:
        st.error(f"Invalid period format: {year} {q}")
        return


    # Build a window that always tries to show exactly window_size quarters
    min_period = gdp_data.index.min()
    max_period = prediction_max_period

    half_window = window_size // 2

    # Start with selected quarter centered
    start_period = selected_period - half_window
    end_period = selected_period + half_window

    # If the right edge goes beyond the max allowed quarter, shift the whole window left
    if end_period > max_period:
        shift = end_period - max_period
        start_period -= shift
        end_period -= shift

    # If the left edge goes before the earliest available quarter, shift the whole window right
    if start_period < min_period:
        shift = min_period - start_period
        start_period += shift
        end_period += shift

    # Final clamp in case the available history is smaller than window_size
    start_period = max(start_period, min_period)
    end_period = min(end_period, max_period)

    full_periods = list(pd.period_range(start=start_period, end=end_period, freq="Q"))

    # If there are still too many or too few due to boundary issues, trim/pad by recomputing from the right
    if len(full_periods) > window_size:
        full_periods = full_periods[:window_size]
    elif len(full_periods) < window_size:
        start_period = max(min_period, end_period - (window_size - 1))
        full_periods = list(pd.period_range(start=start_period, end=end_period, freq="Q"))

    full_labels = [str(p).replace("Q", " Q") for p in full_periods]
    selected_label = str(selected_period).replace("Q", " Q")

    # Pre-load active models from session state (set in config_panel)
    active_models = st.session_state.get("active_models", [])
    valid_nowcasts = nowcasts_df.reindex(full_labels) if not nowcasts_df.empty else pd.DataFrame(index=full_labels)

    # Load Model Data
    ar_preds = load_model_csv("historical_gdp_ar_predictions.csv") if "AR Model" in active_models else pd.Series(dtype=float)
    adl_preds = load_model_csv("historical_gdp_adl_predictions.csv") if "ADL Model" in active_models else pd.Series(dtype=float)
    bridge_preds = load_model_csv("historical_gdp_bridge_predictions.csv") if "Bridge Model" in active_models else pd.Series(dtype=float)

    # Overwrite only the last historical row with live_nowcast_results.csv values
    if "AR Model" in active_models and not ar_preds.empty:
        ar_preds = overwrite_last_row_with_live(ar_preds, live_nowcasts_df, "ar_benchmark")

    if "ADL Model" in active_models and not adl_preds.empty:
        adl_preds = overwrite_last_row_with_live(adl_preds, live_nowcasts_df, "adl_benchmark")

    if "Bridge Model" in active_models and not bridge_preds.empty:
        bridge_live_df = live_nowcasts_df.copy()

        if not bridge_live_df.empty:
            bridge_live_df["bridge_live"] = bridge_live_df["bridge_flash3"]
            bridge_live_df["bridge_live"] = bridge_live_df["bridge_live"].fillna(bridge_live_df["bridge_flash2"])
            bridge_live_df["bridge_live"] = bridge_live_df["bridge_live"].fillna(bridge_live_df["bridge_flash1"])

            bridge_preds = overwrite_last_row_with_live(bridge_preds, bridge_live_df, "bridge_live")

    # PLOTTING
    fig = go.Figure()

    # US Real GDP (Actual)
    if "Real GDP (Actual)" in valid_nowcasts.columns:
        fig.add_trace(go.Scatter(
            x=valid_nowcasts.index, 
            y=valid_nowcasts["Real GDP (Actual)"], 
            mode="lines+markers", 
            name="Actual GDP (FRED)", 
            line=dict(color="#5DADE2", width=4)
        ))

    # Internal Model Mapping
    model_map = [
        ("AR Model", ar_preds, "#E74C3C", "dash", "circle"),
        ("ADL Model", adl_preds, "#9B59B6", "dashdot", "square"),
        ("Bridge Model", bridge_preds, "#F1C40F", "longdash", "diamond")
    ]

    for name, data, color, dash, sym in model_map:
        if name in active_models and not data.empty:
            zoom = data.reindex(full_labels)
            fig.add_trace(go.Scatter(
                x=full_labels, 
                y=zoom.values, 
                mode="lines+markers", 
                name=name, 
                line=dict(dash=dash, width=2, color=color), 
                marker=dict(symbol=sym, size=8)
            ))

    # External Fed Forecasts - Atlanta
    if "Atlanta Fed" in active_models and "Atlanta Fed Forecast" in valid_nowcasts.columns:
        fig.add_trace(go.Scatter(
            x=valid_nowcasts.index, 
            y=valid_nowcasts["Atlanta Fed Forecast"], 
            mode="lines+markers", 
            name="Atlanta Fed (GDPNow)", 
            line=dict(dash="dot", width=3, color="#E67E22")
        ))
        
    # External Fed Forecasts - St. Louis
    if "St. Louis Fed" in active_models and "St. Louis Fed Forecast" in valid_nowcasts.columns:
        fig.add_trace(go.Scatter(
            x=valid_nowcasts.index, 
            y=valid_nowcasts["St. Louis Fed Forecast"], 
            mode="lines+markers", 
            name="St. Louis Fed", 
            line=dict(dash="dot", width=3, color="#2ECC71") # Distinct green styling
        ))

    fig.update_layout(
        title=dict(text=f"Historical Terminal: {year} {q} Focus", font=dict(size=18)),
        template="plotly_dark",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=120, b=40),
        xaxis=dict(gridcolor="#30363d", zeroline=False),
        yaxis=dict(gridcolor="#30363d", title="Annualized Growth (%)")
    )
    fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3)

    if selected_label in full_labels:
        selected_idx = full_labels.index(selected_label)

    fig.add_vrect(
        x0=selected_idx - 0.5,
        x1=selected_idx + 0.5,
        fillcolor="rgba(255, 255, 255, 0.10)",
        line_width=0,
        layer="below"
    )

    fig.add_annotation(
        x=selected_label,
        y=1.08,
        yref="paper",
        text=f"Selected: {selected_label}",
        showarrow=False,
        font=dict(size=12, color="white")
    )
    
    st.plotly_chart(fig, use_container_width=True)