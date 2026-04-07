import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# --- 1. DATA LOADERS ---
@st.cache_data
def load_evolution_data():
    try:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "bridge_evolution.csv"
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_contribution_data():
    try:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "bridge_contributions.csv"
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return pd.DataFrame()

def get_actual_month_labels(quarter_str):
    """Maps a quarter string like '2026 Q1' to its actual calendar months (1-3 only)."""
    q_str = str(quarter_str).replace(" ", "").upper()
    q = q_str[-2:]
    
    if q == "Q1": return ["Jan", "Feb", "Mar"]
    if q == "Q2": return ["Apr", "May", "Jun"]
    if q == "Q3": return ["Jul", "Aug", "Sep"]
    if q == "Q4": return ["Oct", "Nov", "Dec"]
    return ["1st Month", "2nd Month", "3rd Month"]

# Optional: Map FRED tickers to human-readable names
TICKER_MAP = {
    'DPCERA3M086SBEA': 'Personal Consumption',
    'UEMP15T26': 'Unemployment (15-26 wks)',
    'DMANEMP': 'Manufacturing Emp.',
    'IPDMAT': 'Industrial Production',
    'W875RX1': 'Real Personal Income',
    'UNRATE': 'Unemployment Rate',
    'GDP_growth_lag1': 'Lagged GDP (t-1)',
    'GDP_growth_lag2': 'Lagged GDP (t-2)',
    'covid_dummy': 'Covid Shock'
}

# --- 2. SIDEBAR LOGIC ---
def get_sidebar_filters():
    st.subheader("Intra-Quarter Filters")
    df = load_evolution_data()
    
    if df.empty:
        st.info("Awaiting evolution data...")
        return None
        
    quarters = sorted(df['target_quarter'].dropna().unique(), reverse=True)
    default_idx = 0
    if "2025 Q4" in quarters:
        default_idx = quarters.index("2025 Q4")
        
    selected_q = st.selectbox("Target Quarter", quarters, index=default_idx)
    return selected_q

# --- 3. MAIN RENDER ---
def render(gdp_data, selected_q):
    df_evol = load_evolution_data()
    df_contrib = load_contribution_data()
    
    if df_evol.empty or selected_q is None:
        st.warning("Could not load bridge_evolution.csv. Awaiting backend data.")
        return

    st.subheader(f"Bridge Model: Evolution for {selected_q}")

    # Process Data for Line Chart
    q_data = df_evol[df_evol['target_quarter'] == selected_q].copy()
    
    # --- CHANGE 1: Filter out the 4th month ---
    q_data = q_data[q_data['nowcast_month'] <= 3]
    
    month_labels = get_actual_month_labels(selected_q)
    month_map = {1: month_labels[0], 2: month_labels[1], 3: month_labels[2]}
    q_data['month_label'] = q_data['nowcast_month'].map(month_map)
    
    actual_gdp = None
    try:
        q_period = pd.Period(selected_q.replace(" ", ""), freq="Q")
        if q_period in gdp_data.index:
            actual_gdp = gdp_data.loc[q_period]
    except Exception:
        pass

    # Build the Line Chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=q_data['month_label'],
        y=q_data['prediction'],
        mode="lines+markers+text",
        name="Nowcast Evolution",
        line=dict(color="#00E5FF", width=3), 
        marker=dict(size=12, symbol="circle", color="#00E5FF"), 
        text=[f"{v:.2f}%" if pd.notna(v) else "" for v in q_data['prediction']],
        textposition="top center"
    ))
    
    if actual_gdp is not None:
        fig1.add_hline(
            y=actual_gdp, line_dash="dash", line_color="#00FF00", 
            annotation_text=f"Actual: {actual_gdp:.2f}%", 
            annotation_position="bottom right",
            annotation_font=dict(color="#00FF00", size=14)
        )
        
    fig1.update_layout(
        template="plotly_dark",
        xaxis_title="Timeline of Prediction",
        yaxis_title="GDP Growth (%)",
        xaxis=dict(categoryorder="array", categoryarray=month_labels),
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        hovermode="x unified" 
    )
    
    all_vals = q_data['prediction'].dropna().tolist()
    if actual_gdp is not None: all_vals.append(actual_gdp)
    if all_vals:
        min_val, max_val = min(all_vals), max(all_vals)
        padding = max(0.5, abs(max_val - min_val) * 0.4)
        fig1.update_yaxes(range=[min_val - padding, max_val + padding])

    # ---------------------------------------------------------
    # LAYOUT: 5:1 RATIO (Graph : Ranking List)
    # ---------------------------------------------------------
    col_chart, col_list = st.columns([5, 1])

    with col_chart:
        st.plotly_chart(fig1, use_container_width=True)

    with col_list:
        st.markdown("<h5 style='text-align: center; color: #A0AAB5;'>Top Drivers</h5>", unsafe_allow_html=True)
        
        c_data = df_contrib[df_contrib['target_quarter'] == selected_q].copy()
        
        if not c_data.empty and 'nowcast_month' in c_data.columns:
            max_month = c_data['nowcast_month'].max()
            c_data = c_data[c_data['nowcast_month'] == max_month]
        
        if not c_data.empty:
            st.markdown(f"<div style='text-align: center; font-size: 12px; color: #A0AAB5; margin-bottom: 10px;'>Final Drivers for {selected_q}</div>", unsafe_allow_html=True)
            
            c_data['friendly_name'] = c_data['variable'].apply(lambda x: TICKER_MAP.get(x, x))
            c_data = c_data.sort_values(by='contribution', ascending=False)
            
            # --- CHANGE 2: Added max-height and overflow-y to make the list scrollable ---
            list_html = "<div style='font-size: 13px; line-height: 1.6; max-height: 330px; overflow-y: auto; padding-right: 5px;'>"
            
            for _, row in c_data.iterrows():
                val = row['contribution']
                color = "#00FF00" if val > 0 else "#FF3333" 
                sign = "+" if val > 0 else ""
                
                list_html += f"""<div style='margin-bottom: 8px; border-bottom: 1px solid #30363d; padding-bottom: 4px;'>
<span style='color: #fff; font-weight: 500;'>{row['friendly_name']}</span><br>
<span style='color: {color}; font-weight: bold;'>{sign}{val:.2f}%</span>
</div>"""
            list_html += "</div>"
            
            st.markdown(list_html, unsafe_allow_html=True)
        else:
            st.info(f"No driver data found for {selected_q}.")