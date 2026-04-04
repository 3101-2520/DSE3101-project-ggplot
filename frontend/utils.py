import streamlit as st

def apply_custom_font():
    st.markdown(
        """
        <style>
        /* Import the Terminal-style font */
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
        
        /* Apply it globally to all standard Streamlit components */
        html, body, [class*="css"]  {
            font-family: 'IBM Plex Mono', monospace !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )