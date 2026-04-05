import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

@st.cache_data
def load_live_nowcast():
    try:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "live_nowcast_results.csv"
        df = pd.read_csv(csv_path)
        if df.empty: return None
        return df.iloc[0] 
    except Exception:
        return None

def render(show_50=True, show_80=True):
    row = load_live_nowcast()
    
    if row is None:
        st.info("Awaiting live nowcast results from backend...")
        return

    quarter = str(row['quarter']).strip()
    q_str = quarter[-2:]
    
    month_map = {
        "Q1": ["January", "February", "March"],
        "Q2": ["April", "May", "June"],
        "Q3": ["July", "August", "September"],
        "Q4": ["October", "November", "December"]
    }
    m_labels = month_map.get(q_str, ["1st Month", "2nd Month", "3rd Month"])

    # Smart multiplier
    multiplier = 100 if abs(row.get('ar_benchmark', 0)) < 0.5 else 1

    months = []
    preds = []
    ses = [] 
    
    # Load predictions AND standard errors
    if pd.notna(row.get('bridge_flash1')):
        months.append(m_labels[0])
        preds.append(row['bridge_flash1'] * multiplier)
        ses.append(row.get('bridge_flash1_se', 0.009) * multiplier)
        
    if pd.notna(row.get('bridge_flash2')):
        months.append(m_labels[1])
        preds.append(row['bridge_flash2'] * multiplier)
        ses.append(row.get('bridge_flash2_se', 0.005) * multiplier)
        
    if pd.notna(row.get('bridge_flash3')):
        months.append(m_labels[2])
        preds.append(row['bridge_flash3'] * multiplier)
        ses.append(row.get('bridge_flash3_se', 0.002) * multiplier)

    st.markdown(f"<h4 style='color: white; margin-bottom: 0px;'>Live Bridge Nowcast Evolution ({quarter})</h4>", unsafe_allow_html=True)

    fig = go.Figure()

    if preds:
        # --- FAN CHART LOGIC ---
        z_80, z_50 = 1.28, 0.67
        
        if len(ses) == len(preds):
            upper_80 = [p + (se * z_80) for p, se in zip(preds, ses)]
            lower_80 = [p - (se * z_80) for p, se in zip(preds, ses)]
            upper_50 = [p + (se * z_50) for p, se in zip(preds, ses)]
            lower_50 = [p - (se * z_50) for p, se in zip(preds, ses)]

            # 80% Interval
            if show_80:
                fig.add_trace(go.Scatter(
                    x=months + months[::-1], 
                    y=upper_80 + lower_80[::-1],
                    fill='toself',
                    fillcolor='rgba(0, 229, 255, 0.1)', 
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    name="80% Interval",
                    showlegend=False
                ))

            # 50% Interval
            if show_50:
                fig.add_trace(go.Scatter(
                    x=months + months[::-1],
                    y=upper_50 + lower_50[::-1],
                    fill='toself',
                    fillcolor='rgba(0, 229, 255, 0.2)', 
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    name="50% Interval",
                    showlegend=False
                ))

        # --- CONDITIONAL TEXT LABELS ---
        # If BOTH intervals are turned off, show text on the previous months
        if not show_50 and not show_80:
            track_mode = "lines+markers+text"
            # Label all points EXCEPT the last one (which has its own big label)
            track_text = [f"{p:.2f}%" for p in preds[:-1]] + [""]
        else:
            track_mode = "lines+markers"
            track_text = None

        # 1. The tracking line (Teal)
        fig.add_trace(go.Scatter(
            x=months,
            y=preds,
            mode=track_mode,
            text=track_text,
            textposition="top center",
            textfont=dict(color="#00E5FF", size=13, family="Arial"),
            name="Evolution Track",
            line=dict(color="#00E5FF", width=3),
            marker=dict(size=10, color="#00E5FF"),
            hovertemplate="<b>%{x}</b><br>Prediction: %{y:.2f}%<extra></extra>",
            hoverlabel=dict(bgcolor="#1e2127", font=dict(color="white", size=14), bordercolor="#30363d"),
            showlegend=False
        ))

        # 2. Latest Flash Marker (Upgraded to Emerald Green)
        latest_month = months[-1]
        latest_pred = preds[-1]
        
        emerald_green = "#2ECC71"

        fig.add_trace(go.Scatter(
            x=[latest_month],
            y=[latest_pred],
            mode="markers+text",
            name="Current Nowcast",
            marker=dict(size=22, color=emerald_green, symbol="circle", line=dict(color="white", width=2)),
            text=[f"{latest_pred:.2f}%"],
            textposition="top center",
            textfont=dict(size=16, color=emerald_green, family="Arial Black"),
            hovertemplate="<b>Current Nowcast</b><br>%{x}<br>Value: %{y:.2f}%<extra></extra>",
            hoverlabel=dict(bgcolor="#1e2127", font=dict(color=emerald_green, size=15), bordercolor=emerald_green),
            showlegend=False
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Time of Prediction",
        yaxis_title="Predicted GDP Growth (%)",
        xaxis=dict(categoryorder="array", categoryarray=m_labels),
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380
    )

    # Smart Y-axis scaling
    if preds:
        min_val, max_val = min(preds), max(preds)
        
        if len(ses) == len(preds):
            if show_80:
                min_val, max_val = min(lower_80), max(upper_80)
            elif show_50:
                min_val, max_val = min(lower_50), max(upper_50)
                
        padding = max(0.5, abs(max_val - min_val) * 0.3)
        fig.update_yaxes(range=[min_val - padding, max_val + padding])

    st.plotly_chart(fig, use_container_width=True)