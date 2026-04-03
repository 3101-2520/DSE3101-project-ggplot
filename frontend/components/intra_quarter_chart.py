import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

@st.cache_data
def load_evolution_data():
    try:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "bridge_evolution.csv"
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return pd.DataFrame()

def render(gdp_data):
    st.subheader("Bridge Model: Intra-Quarter Evolution")
    
    df = load_evolution_data()
    
    if df.empty:
        st.warning("Could not load bridge_evolution.csv. Awaiting backend data.")
        return

    # 1. Dropdown for Quarters (Sorted newest to oldest)
    quarters = sorted(df['target_quarter'].dropna().unique(), reverse=True)
    selected_q = st.selectbox("Select Target Quarter to Track", quarters)
    
    # 2. Filter data for the selected quarter
    q_data = df[df['target_quarter'] == selected_q].copy()
    
    # Map numeric months to readable labels
    month_map = {1: "1st Month", 2: "2nd Month", 3: "3rd Month", 4: "Month After"}
    q_data['month_label'] = q_data['nowcast_month'].map(month_map)
    
    # 3. Look up the ACTUAL GDP for this quarter (if it exists yet!)
    actual_gdp = None
    try:
        q_period = pd.Period(selected_q.replace(" ", ""), freq="Q")
        if q_period in gdp_data.index:
            actual_gdp = gdp_data.loc[q_period]
    except Exception:
        pass

    # 4. Build the Chart
    fig = go.Figure()
    
    # The Evolution Line (Sleek Cyan)
    fig.add_trace(go.Scatter(
        x=q_data['month_label'],
        y=q_data['prediction'],
        mode="lines+markers+text",
        name="Nowcast Evolution",
        line=dict(color="#00E5FF", width=3), 
        marker=dict(size=12, symbol="circle", color="#00E5FF"),
        text=[f"{v:.2f}%" if pd.notna(v) else "" for v in q_data['prediction']],
        textposition="top center"
    ))
    
    # The Actual GDP Target Line (Neon Green)
    if actual_gdp is not None:
        fig.add_hline(
            y=actual_gdp, 
            line_dash="dash", 
            line_color="#00FF00", 
            annotation_text=f"Actual GDP Print: {actual_gdp:.2f}%", 
            annotation_position="bottom right",
            annotation_font=dict(color="#00FF00", size=14)
        )
        
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Time of Prediction",
        yaxis_title="Predicted GDP Growth (%)",
        xaxis=dict(
            categoryorder="array", 
            categoryarray=["1st Month", "2nd Month", "3rd Month", "Month After"]
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400
    )
    
    # Smart Y-axis scaling to make sure both the line and target fit nicely
    all_vals = q_data['prediction'].dropna().tolist()
    if actual_gdp is not None:
        all_vals.append(actual_gdp)
    
    if all_vals:
        min_val, max_val = min(all_vals), max(all_vals)
        padding = max(0.5, abs(max_val - min_val) * 0.3)
        fig.update_yaxes(range=[min_val - padding, max_val + padding])

    st.plotly_chart(fig, use_container_width=True)