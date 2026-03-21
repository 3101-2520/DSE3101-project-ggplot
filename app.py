import streamlit as st 

st.title("DSE3101 Project Dashboard")

st.subheader("Simple Time-Series Plot")
# Generating some dummy data (simulating economic indicators)
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['Indicator A', 'Indicator B', 'Indicator C']
)

# This creates an interactive line chart automatically
st.line_chart(chart_data)

st.write("You can toggle sidebar options or add sliders to filter this data later!")

