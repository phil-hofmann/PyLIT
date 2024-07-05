import streamlit as st
from pylit import ui


def main():
    ui.components.settings_manager()

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
    st.session_state["view_coeffs_plot"] = st.toggle(
        "Show Coefficients Plot", value=st.session_state["view_coeffs_plot"]
    )
    st.session_state["wide_mode"] = st.toggle(
        "Wide Mode", value=st.session_state["wide_mode"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save Settings"):
            ui.utils.save_settings(
                ui.settings.PATH_SETTINGS,
                ui.utils.load_settings(st.session_state),
            )
            st.toast("Settings saved successfully.", icon="✅")

    with col2:
        if st.button("Reset Settings"):
            ui.components.reset_settings()
            ui.utils.save_settings(
                ui.settings.PATH_SETTINGS, ui.utils.load_settings(st.session_state)
            )
            st.rerun()


if __name__ == "__main__":
    main()
