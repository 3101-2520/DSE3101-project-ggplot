import streamlit as st
import pandas as pd
from fredapi import Fred
import plotly.graph_objects as go

# Setup
try:
    api_key = st.secrets["FRED_API_KEY"]
    fred = Fred(api_key=api_key)
except KeyError:
    st.error("FRED_API_KEY not found! Please check your secrets configuration.")

@st.cache_data(ttl=3600)
def fetch_nowcast_comparison():
    # Atlanta Fed: Still the gold standard 'GDPNOW'
    # St. Louis Fed: 'STLENI' (Excellent, stable alternative to the NY Fed)
    # Realized GDP: 'A191RL1Q225SBEA' (For historical context)
    series_map = {
        'Atlanta Fed (GDPNow)': 'GDPNOW',
        'St. Louis Fed (Nowcast)': 'STLENI'
    }
    
    combined_df = pd.DataFrame()
    
    for label, series_id in series_map.items():
        try:
            series_data = fred.get_series(series_id)
            # We take only the last 180 days to keep the 'current' quarter focused
            combined_df[label] = series_data.tail(180)
        except Exception as e:
            st.warning(f"Could not fetch {label}: {e}")
            
    return combined_df.sort_index().ffill()

st.subheader("🏦 Federal Reserve Nowcast Comparison")

df = fetch_nowcast_comparison()

if not df.empty:
    # Plotly for a high-end dashboard feel
    fig = go.Figure()
    
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[col], 
            mode='lines+markers', 
            name=col,
            hovertemplate='%{x|%b %d, %Y}<br>Estimate: %{y:.2f}%'
        ))

    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Release Date",
        yaxis_title="Annualized Real GDP Growth (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Show latest readings as metrics
    cols = st.columns(len(df.columns))
    for i, name in enumerate(df.columns):
        latest_val = df[name].iloc[-1]
        prev_val = df[name].iloc[-2] if len(df) > 1 else latest_val
        cols[i].metric(name, f"{latest_val:.2f}%", f"{latest_val - prev_val:.2f}%")
else:
    st.error("No data found. Check your FRED API key and Series IDs.")
