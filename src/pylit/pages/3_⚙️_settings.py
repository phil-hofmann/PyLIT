import pylit
import streamlit as st

def main():
    # NOTE deprecated pylit.ui.components.state_manager()
    pylit.ui.components.settings_manager()

    # Page settings
    st.set_page_config(
        page_title="Experiment",
        page_icon="🚀",
        layout="wide" if st.session_state["wide_mode"] else "centered",
    )
    st.subheader("Settings")
    st.session_state["workspace"] = st.text_input("Workspace", value=st.session_state["workspace"])
    st.session_state["force"] = st.toggle("Force Run Experiment", value=st.session_state["force"])
    st.session_state["interactive"] = st.toggle("Interactive Plotting", value=st.session_state["interactive"])
    st.session_state["sliders"] = st.toggle("Use Sliders", value=st.session_state["sliders"])
    st.session_state["store"] = st.toggle("Store Experiment", value=st.session_state["store"])
    st.session_state["docx"] = st.toggle("Create Report", value=st.session_state["docx"])
    st.session_state["view_config_json"] = st.toggle("Show Config JSON", value=st.session_state["view_config_json"])
    st.session_state["view_models_json"] = st.toggle("Show Models JSON", value=st.session_state["view_models_json"])
    st.session_state["view_report"] = st.toggle("Show Report", value=st.session_state["view_report"])
    st.session_state["view_coeffs_plot"] = st.toggle("Show Coefficients Plot", value=st.session_state["view_coeffs_plot"])
    st.session_state["wide_mode"] = st.toggle("Wide Mode", value=st.session_state["wide_mode"])

    col1, col2 = st.columns(2)


    with col1:
        if st.button("Save Settings"):
            pylit.ui.utils.save_settings(
                pylit.ui.settings.PATH_SETTINGS,
                pylit.ui.utils.load_settings(st.session_state),
            )
            st.toast("Settings saved successfully.", icon="✅")
    
    with col2:
        if st.button("Reset Settings"):
            pylit.ui.components.reset_settings()
            pylit.ui.utils.save_settings(
                pylit.ui.settings.PATH_SETTINGS,
                pylit.ui.utils.load_settings(
                    st.session_state
                )
            )
            st.rerun()

if __name__ == "__main__":
    main()