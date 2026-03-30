import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

@st.cache_data
def load_evolution_data():
    try:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "bridge_evolution.csv"
        df = pd.read_csv(csv_path)
        df['target_quarter'] = df['target_quarter'].astype(str).str.replace(" ", "")
        df['target_quarter'] = df['target_quarter'].str.replace("Q", " Q")
        return df
    except Exception as e:
        return pd.DataFrame()

def render():
    st.subheader("Bridge Model: Intra-Quarter Evolution")
    df = load_evolution_data()
    
    if df.empty:
        st.info("Waiting for backend to generate bridge_evolution.csv...")
        return

    available_quarters = sorted(df['target_quarter'].unique(), reverse=True)
    selected_q = st.selectbox("Select Target Quarter to Track", available_quarters, key="evo_q_select")

    q_data = df[df['target_quarter'] == selected_q].copy()
    month_labels = {1: "1st Month", 2: "2nd Month", 3: "3rd Month", 4: "Month After"}
    q_data['month_label'] = q_data['nowcast_month'].map(month_labels)
    q_data = q_data.sort_values('nowcast_month')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=q_data['month_label'],
        y=q_data['prediction'], # <-- No multiplier
        mode="lines+markers+text",
        name="Bridge Nowcast",
        line=dict(color="#F1C40F", width=3),
        marker=dict(size=10, symbol="diamond"),
        text=q_data['prediction'].round(2).astype(str) + "%", # <-- No multiplier
        textposition="top center",
        connectgaps=False
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)

    fig.update_layout(
        title=f"Evolution of GDP Nowcast for {selected_q}",
        template="plotly_dark",
        xaxis_title="Time of Prediction",
        yaxis_title="Predicted GDP Growth (%)",
        xaxis=dict(categoryorder="array", categoryarray=["1st Month", "2nd Month", "3rd Month", "Month After"]),
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)