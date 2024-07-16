import streamlit as st
from pylit.frontend.utils import (
    settings_manager,
    save_settings,
    load_settings,
    reset_settings,
)
from pylit.frontend.constants import PATH_SETTINGS


def main():
    settings_manager()

    # Page settings
    st.set_page_config(
        page_title="Experiment",
        page_icon="🚀",
        layout="wide" if st.session_state["wide_mode"] else "centered",
    )
    st.subheader("Settings")
    st.session_state["workspace"] = st.text_input(
        "Workspace", value=st.session_state["workspace"]
    )
    st.session_state["wide_mode"] = st.toggle(
        "Wide Mode", value=st.session_state["wide_mode"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save Settings"):
            save_settings(
                PATH_SETTINGS,
                load_settings(st.session_state),
            )
            st.toast("Settings saved successfully.", icon="✅")

    with col2:
        if st.button("Reset Settings"):
            reset_settings()
            save_settings(PATH_SETTINGS, load_settings(st.session_state))
            st.rerun()


if __name__ == "__main__":
    main()
