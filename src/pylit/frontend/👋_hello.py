from pylit.frontend.utils import settings_manager
import streamlit as st

def main():
    settings_manager()

    # Page settings
    st.set_page_config(
        page_title="Welcome!",
        page_icon="🚀",
        layout="wide" if st.session_state["wide_mode"] else "centered",
    )

    st.title("Welcome to Pylit 🚀!")

if __name__ == "__main__":
    main()