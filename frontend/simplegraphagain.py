import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Economic Nowcasting Dashboard")

# Load data
df = pd.read_csv("data/data.csv")

# Sidebar filters
indicator = st.sidebar.selectbox("Select Indicator", df['indicator'].unique())

filtered = df[df['indicator'] == indicator]

# Plot
fig = px.line(filtered, x='date', y='value', title=indicator)
st.plotly_chart(fig)

# Show data
st.dataframe(filtered)
