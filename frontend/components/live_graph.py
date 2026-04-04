import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# --- 1. DATA LOADER ---
@st.cache_data
def load_live_nowcast():
    try:
        # Resolves path to data/live_nowcast_results.csv
        csv_path = Path(__file__).resolve().parents[2] / "data" / "live_nowcast_results.csv"
        df = pd.read_csv(csv_path)
        if df.empty: return None
        return df.iloc[0] 
    except Exception:
        return None

# --- 2. MAIN RENDER ---
def render():
    row = load_live_nowcast()
    
    if row is None:
        st.info("Awaiting live nowcast results from backend...")
        return

    quarter = row['quarter']
    q_str = quarter[-2:]
    
    # Map quarters to months for the X-axis
    month_map = {
        "Q1": ["January", "February", "March"],
        "Q2": ["April", "May", "June"],
        "Q3": ["July", "August", "September"],
        "Q4": ["October", "November", "December"]
    }
    m_labels = month_map.get(q_str, ["1st Month", "2nd Month", "3rd Month"])

    # Scale check: ensures values are in percentage format
    multiplier = 100 if abs(row.get('ar_benchmark', 0)) < 0.5 else 1

    months = []
    preds = []
    
    # Populate available predictions from the Bridge model flashes
    if pd.notna(row.get('bridge_flash1')):
        months.append(m_labels[0])
        preds.append(row['bridge_flash1'] * multiplier)
    if pd.notna(row.get('bridge_flash2')):
        months.append(m_labels[1])
        preds.append(row['bridge_flash2'] * multiplier)
    if pd.notna(row.get('bridge_flash3')):
        months.append(m_labels[2])
        preds.append(row['bridge_flash3'] * multiplier)

    fig = go.Figure()

    if preds:
        # 1. Evolution Track (Cyan line for historical trend in current quarter)
        fig.add_trace(go.Scatter(
            x=months,
            y=preds,
            mode="lines+markers+text", 
            name="Evolution Track",
            line=dict(color="#00E5FF", width=3),
            marker=dict(size=10, color="#00E5FF"),
            # Labels for historical months in the current quarter
            text=[f"{v:.2f}%" if i < len(preds)-1 else "" for i, v in enumerate(preds)],
            textposition="top center",
            textfont=dict(size=12, color="#00E5FF"),
            hoverinfo="skip",
            showlegend=False
        ))

        # 2. Latest Nowcast Highlight
        latest_month = months[-1]
        latest_pred = preds[-1]
        
        fig.add_trace(go.Scatter(
            x=[latest_month],
            y=[latest_pred],
            mode="markers+text",
            name="Current Nowcast",
            marker=dict(
                size=16, 
                color="#00E5FF", 
                symbol="circle", 
                line=dict(color="white", width=2)
            ),
            text=[f"{latest_pred:.2f}%"],
            textposition="top center",
            textfont=dict(
                size=16, 
                color="#00E5FF"
            ),
            showlegend=False
        ))

    # Reference zero-line
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)

    # --- FIXED LAYOUT LOGIC ---
    fig.update_layout(
        title=dict(
            text=f"Live Bridge Nowcast Evolution ({quarter})",
            font=dict(size=16, color="white")
        ),
        template="plotly_dark",
        xaxis_title="Timeline of Prediction",
        yaxis_title="Predicted GDP Growth (%)",
        xaxis=dict(
            categoryorder="array", 
            categoryarray=m_labels
        ), 
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350
    )

    if preds:
        min_val, max_val = min(preds), max(preds)
        padding = max(0.8, abs(max_val - min_val) * 0.8) 
        fig.update_yaxes(range=[min_val - padding, max_val + padding])

    st.plotly_chart(fig, use_container_width=True)