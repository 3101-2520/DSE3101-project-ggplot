# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# from .atlanta_fed import get_historical_nowcasts

# def render(gdp_growth):
#     # Fetch historical nowcasts
#     nowcasts_df = get_historical_nowcasts()

#     st.markdown("### Chart Controls")
#     col1, col2 = st.columns(2)

#     min_year = int(gdp_growth.index.min().year)
#     max_year = int(gdp_growth.index.max().year)

#     with col1:
#         year = st.number_input(
#             "Select Year",
#             min_value=min_year,
#             max_value=max_year,
#             value=2024,
#             step=1
#         )
#     with col2:
#         q = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])

#     selected_period = pd.Period(f"{year}{q}", freq="Q")

#     if selected_period not in gdp_growth.index:
#         st.warning(f"Selected period {year} {q} is not in your dataset yet.")
#         return

#     # Keep exactly 7 quarters when possible
#     idx = gdp_growth.index.get_loc(selected_period)
#     start = idx - 3
#     end = idx + 3

#     if start < 0:
#         end += -start
#         start = 0

#     if end > len(gdp_growth) - 1:
#         shift_left = end - (len(gdp_growth) - 1)
#         start -= shift_left
#         end = len(gdp_growth) - 1

#     start = max(0, start)

#     hist_zoom = (gdp_growth.iloc[start:end + 1] * 100).to_frame(name="Growth")
#     hist_zoom["label"] = hist_zoom.index.astype(str).str.replace("Q", " Q", regex=False)

#     all_x_labels = hist_zoom["label"].tolist()

#     # Match nowcast rows to the same 7 displayed quarters
#     if not nowcasts_df.empty:
#         valid_nowcasts = nowcasts_df[nowcasts_df.index.isin(all_x_labels)].copy()
#     else:
#         valid_nowcasts = pd.DataFrame()

#     fig = go.Figure()

#     # Actual GDP
#     fig.add_trace(go.Scatter(
#         x=hist_zoom["label"],
#         y=hist_zoom["Growth"],
#         mode="lines+markers",
#         name="Actual GDP",
#         line=dict(color="#5DADE2", width=4),
#         marker=dict(size=8)
#     ))

#     active_models = st.session_state.get("active_models", [])

#     if not valid_nowcasts.empty:
#         # Atlanta Fed checkbox name -> DataFrame column name
#         if "Atlanta Fed" in active_models and "Atlanta Fed Forecast" in valid_nowcasts.columns:
#             atl_data = valid_nowcasts["Atlanta Fed Forecast"].dropna()
#             if not atl_data.empty:
#                 fig.add_trace(go.Scatter(
#                     x=atl_data.index,
#                     y=atl_data.values,
#                     mode="lines+markers",
#                     name="Atlanta Fed (GDPNow)",
#                     line=dict(dash="dot", width=3, color="#E67E22"),
#                     marker=dict(size=8)
#                 ))

#         # St. Louis checkbox name -> DataFrame column name
#         if "St. Louis Fed" in active_models and "St. Louis Fed Forecast" in valid_nowcasts.columns:
#             stl_data = valid_nowcasts["St. Louis Fed Forecast"].dropna()
#             if not stl_data.empty:
#                 fig.add_trace(go.Scatter(
#                     x=stl_data.index,
#                     y=stl_data.values,
#                     mode="lines+markers",
#                     name="St. Louis Fed Forecast",
#                     line=dict(dash="dot", width=3, color="#58D68D"),
#                     marker=dict(size=8)
#                 ))

#     fig.update_layout(
#         title=f"GDP Growth & Forecasts around {year} {q}",
#         template="plotly_dark",
#         hovermode="x unified",
#         xaxis_title="Quarter",
#         yaxis_title="GDP Growth (%)",
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1
#         ),
#         margin=dict(l=0, r=0, t=50, b=0),
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)"
#     )

#     fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
#     fig.update_xaxes(categoryorder="array", categoryarray=all_x_labels)

#     st.plotly_chart(fig, use_container_width=True)


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .atlanta_fed import get_historical_nowcasts

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