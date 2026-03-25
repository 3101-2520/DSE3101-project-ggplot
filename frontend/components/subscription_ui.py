import streamlit as st
# Simple relative import since they are in the same directory
try:
    from .email_logic import save_subscriber
except ImportError:
    from email_logic import save_subscriber

def render():
    st.markdown("### 📬 Monthly Nowcast Newsletter")
    with st.form("newsletter_form", clear_on_submit=True):
        email = st.text_input("Subscribe for monthly GDP insights", placeholder="email@example.com")
        submit = st.form_submit_button("Join Mailing List")
        
        if submit:
            if email:
                success, message = save_subscriber(email)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please enter an email.")