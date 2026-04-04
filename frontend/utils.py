import streamlit as st

def apply_custom_font():
    st.markdown(
        """
        <style>
        /* Import Inter with Regular (400), Bold (700), and Black (900) weights */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
        
        /* Apply to all standard text */
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif !important;
        }

        /* Force headers to be massive and slightly squished, mimicking Apple's SF Pro Display */
        h1, h2, h3, h4, h5, h6, span {
            font-family: 'Inter', sans-serif !important;
            font-weight: 900 !important; 
            letter-spacing: -0.02em !important; 
        }

        /* Make sure tabs also get the clean font */
        button[data-baseweb="tab"] {
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )