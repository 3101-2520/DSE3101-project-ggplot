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
        return df.iloc[-1] 
    except Exception:
        return None

def get_month_labels(quarter_str):
    """Converts a quarter string like '2026 Q1' into specific month/year labels."""
    q_str = str(quarter_str).replace(" ", "").upper()
    year = q_str[:4] if len(q_str) >= 4 else "XX"
    short_year = year[-2:]
    q = q_str[-2:]
    
    month_map = {
        "Q1": ["Jan", "Feb", "Mar"],
        "Q2": ["Apr", "May", "Jun"],
        "Q3": ["Jul", "Aug", "Sep"],
        "Q4": ["Oct", "Nov", "Dec"]
    }
    months = month_map.get(q, ["M1", "M2", "M3"])
    return [f"{m} '{short_year}" for m in months]

def render(show_50=False, show_80=False):
    row = load_live_nowcast()
    
    if row is None:
        st.info("Awaiting live nowcast results from backend...")
        return

    live_quarter = str(row['quarter']).strip()
    live_labels = get_month_labels(live_quarter)
    
    months = []
    preds = []
    ses = []

    multiplier = 100 if abs(row.get('ar_benchmark', 0)) < 0.5 else 1
    
    if pd.notna(row.get('bridge_flash1')):
        months.append(live_labels[0])
        preds.append(row['bridge_flash1'] * multiplier)
        ses.append(row.get('bridge_flash1_se', 0.009) * multiplier)
        
    if pd.notna(row.get('bridge_flash2')):
        months.append(live_labels[1])
        preds.append(row['bridge_flash2'] * multiplier)
        ses.append(row.get('bridge_flash2_se', 0.005) * multiplier)
        
    if pd.notna(row.get('bridge_flash3')):
        months.append(live_labels[2])
        preds.append(row['bridge_flash3'] * multiplier)
        ses.append(row.get('bridge_flash3_se', 0.002) * multiplier)

    if not preds:
        st.info(f"No prediction data available for {live_quarter}.")
        return

    st.markdown(f"<h4 style='color: white; margin-bottom: 0px;'>Live Bridge Nowcast Evolution ({live_quarter})</h4>", unsafe_allow_html=True)

    fig = go.Figure()

    # --- FAN CHART & ERROR BAR LOGIC ---
    z_80, z_50 = 1.28, 0.67
    hover_data = []

    if len(ses) == len(preds):
        upper_80 = [p + (se * z_80) for p, se in zip(preds, ses)]
        lower_80 = [p - (se * z_80) for p, se in zip(preds, ses)]
        upper_50 = [p + (se * z_50) for p, se in zip(preds, ses)]
        lower_50 = [p - (se * z_50) for p, se in zip(preds, ses)]

        if len(preds) > 1:
            # Draw standard Fan Chart bands if we have 2+ points
            if show_80:
                fig.add_trace(go.Scatter(
                    x=months + months[::-1], 
                    y=upper_80 + lower_80[::-1],
                    fill='toself',
                    fillcolor='rgba(0, 229, 255, 0.1)', 
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ))
            if show_50:
                fig.add_trace(go.Scatter(
                    x=months + months[::-1],
                    y=upper_50 + lower_50[::-1],
                    fill='toself',
                    fillcolor='rgba(0, 229, 255, 0.2)', 
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ))
        else:
            # Draw Error Bars if we only have 1 point
            if show_80:
                fig.add_trace(go.Scatter(
                    x=[months[0]], y=[preds[0]], mode='markers',
                    marker=dict(size=0.1, color='rgba(0,0,0,0)'),
                    error_y=dict(
                        type='data', symmetric=True, array=[ses[0] * z_80],
                        color='rgba(0, 229, 255, 0.3)', thickness=2, width=15
                    ),
                    hoverinfo='skip', showlegend=False
                ))
            if show_50:
                fig.add_trace(go.Scatter(
                    x=[months[0]], y=[preds[0]], mode='markers',
                    marker=dict(size=0.1, color='rgba(0,0,0,0)'),
                    error_y=dict(
                        type='data', symmetric=True, array=[ses[0] * z_50],
                        color='rgba(0, 229, 255, 0.6)', thickness=4, width=10
                    ),
                    hoverinfo='skip', showlegend=False
                ))

        for i in range(len(preds)):
            htext = f"Prediction: {preds[i]:.2f}%"
            if show_50: htext += f"<br>50% Range: [{lower_50[i]:.2f}%, {upper_50[i]:.2f}%]"
            if show_80: htext += f"<br>80% Range: [{lower_80[i]:.2f}%, {upper_80[i]:.2f}%]"
            hover_data.append(htext)
    else:
        hover_data = [f"Prediction: {p:.2f}%" for p in preds]

    # --- MAIN TRACKING LINE ---
    track_mode = "lines+markers+text" if len(preds) > 1 else "markers+text"
    track_text = [f"{p:.2f}%" for p in preds[:-1]] + [""]

    fig.add_trace(go.Scatter(
        x=months,
        y=preds,
        mode=track_mode,
        text=track_text,
        hovertext=hover_data,
        textposition="top center",
        textfont=dict(color="#00E5FF", size=13, family="Arial"),
        name="Evolution Track",
        line=dict(color="#00E5FF", width=3),
        marker=dict(size=10, color="#00E5FF"),
        hovertemplate="<b>%{x}</b><br>%{hovertext}<extra></extra>",
        hoverlabel=dict(bgcolor="#1e2127", font=dict(color="white", size=14), bordercolor="#30363d"),
        showlegend=False
    ))

    # --- LATEST FLASH OVERLAY (Emerald Green) ---
    latest_month = months[-1]
    latest_pred = preds[-1]
    emerald_green = "#2ECC71"

    fig.add_trace(go.Scatter(
        x=[latest_month],
        y=[latest_pred],
        mode="markers+text",
        name="Current Nowcast",
        hovertext=[hover_data[-1]], 
        marker=dict(size=22, color=emerald_green, symbol="circle", line=dict(color="white", width=2)),
        text=[f"{latest_pred:.2f}%"],
        textposition="top center",
        textfont=dict(size=16, color=emerald_green, family="Arial Black"),
        hovertemplate="<b>Current Nowcast</b><br>%{x}<br>%{hovertext}<extra></extra>",
        hoverlabel=dict(bgcolor="#1e2127", font=dict(color="white", size=14), bordercolor=emerald_green),
        showlegend=False
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Timeline",
        yaxis_title="Predicted GDP Growth (%)",
        xaxis=dict(
            categoryorder="array", 
            categoryarray=live_labels, 
            range=[-0.2, 2.2] 
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380
    )

    if preds:
        min_val, max_val = min(preds), max(preds)
        
        if len(ses) == len(preds):
            if show_80:
                min_val, max_val = min(lower_80), max(upper_80)
            elif show_50:
                min_val, max_val = min(lower_50), max(upper_50)
                
        padding = max(0.5, abs(max_val - min_val) * 0.5) 
        
        if len(preds) == 1:
            fig.update_yaxes(range=[min_val - padding, max_val + padding])
        else:
            fig.update_yaxes(range=[min_val - padding, max_val + padding])

    st.plotly_chart(fig, use_container_width=True)