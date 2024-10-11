import os
import streamlit as st
from pylit.frontend.utils import (
    settings_manager,
    save_settings,
    load_settings,
    reset_settings,
)
from pylit.frontend.constants import PATH_SETTINGS

# Settings:
# - workspace_as: str
# - wide_mode_as: bool
# - auto_expand_as: bool
# - non_negative_as: bool
# - detailed_balance_as: bool
# - normalization_as: bool
# - time_scaling_as: bool
# - default_model_as: bool
# - forward_default_model_as: bool
# - forward_default_model_error_as: bool
# - noise_samples_as: bool
# - coefficients_as: bool
# - model_as: bool
# - forward_model_as: bool
# - model_error_as: bool
# - forward_model_error_as: bool


def main():
    settings_manager()

    # Page settings
    st.set_page_config(
        page_title="Experiment",
        page_icon="🚀",
        layout="wide" if st.session_state["wide_mode_as"] else "centered",
    )

    st.title("Settings")

    with st.expander("General", expanded=st.session_state["auto_expand_as"]):
        col = st.columns([1, 2], vertical_alignment="center")
        with col[0]:
            st.markdown("**Workspace Directory**")
        with col[1]:
            st.session_state["workspace_as"] = st.text_input(
                label="",
                value=st.session_state["workspace_as"],
                key="workspace_as_input",
            )
        if not os.path.exists(st.session_state["workspace_as"]):
            st.error(
                f'The directory "{st.session_state["workspace_as"]}" does not exist.'
            )
        else:
            pass

        col = st.columns([1, 2], vertical_alignment="center")
        with col[0]:
            st.session_state["wide_mode_as"] = st.toggle(
                label="Wide Mode",
                value=st.session_state["wide_mode_as"],
                key="wide_mode_as_toggle",
            )
        with col[1]:
            st.write("Displays the interface in a wide layout.")

        col = st.columns([1, 2], vertical_alignment="center")
        with col[0]:
            st.session_state["auto_expand_as"] = st.toggle(
                label="Auto Expand",
                value=st.session_state["auto_expand_as"],
                key="auto_expand_as_toggle",
            )
        with col[1]:
            st.write("Automatically expands expanders.")

    with st.expander("Data Adjustments", expanded=st.session_state["auto_expand_as"]):

        col = st.columns([1, 2], vertical_alignment="center")
        with col[0]:
            st.session_state["non_negative_as"] = st.toggle(
                label="Non-Negativity",
                value=st.session_state["non_negative_as"],
                key="non_negative_as_toggle",
            )
        with col[1]:
            st.write("Ensures that all values of the input data are non-negative.")

        col = st.columns([1, 2], vertical_alignment="center")
        with col[0]:
            st.session_state["detailed_balance_as"] = st.toggle(
                label="Detailed Balance",
                value=st.session_state["detailed_balance_as"],
                key="detailed_balance_as_toggle",
            )
        with col[1]:
            st.write(
                "Ensures detailed balance in the data and supplements it as needed."
            )

    with st.expander("Hidden Adjustments", expanded=st.session_state["auto_expand_as"]):
        col = st.columns([1, 2], vertical_alignment="center")
        with col[0]:
            st.session_state["normalization_as"] = st.toggle(
                label="Normalization",
                value=st.session_state["normalization_as"],
                key="normalization_as_toggle",
            )
        with col[1]:
            st.write(
                "Normalizes the data of F to its maximum value and the data of the default model S to its integral, using the trapezoidal rule."
            )

        col = st.columns([1, 2], vertical_alignment="center")
        with col[0]:
            st.session_state["time_scaling_as"] = st.toggle(
                label="Time Scaling",
                value=st.session_state["time_scaling_as"],
                key="time_scaling_as_toggle",
            )
        with col[1]:
            st.write(
                "Scales the time axis of the input data tau to the standard interval [0, 1] and transforms the model output accordingly."
            )

    with st.expander("Description Plotting Options", expanded=st.session_state["auto_expand_as"]):
        col = st.columns([1, 2], vertical_alignment="center")
        with col[0]:
            st.session_state["default_model_as"] = st.toggle(
                label="Default Model",
                value=st.session_state["default_model_as"],
                key="default_model_as_toggle",
            )
        with col[1]:
            st.write("Visualizes the default model D(ω)")
        
        col = st.columns([1, 2], vertical_alignment="center")
        with col[0]:
            st.session_state["forward_default_model_as"] = st.toggle(
                label="Forward D. Model",
                value=st.session_state["forward_default_model_as"],
                key="forward_default_model_as_toggle",
            )
        with col[1]:
            st.write("Visualizes the forward default model L(D)(τ).")
        
        col = st.columns([1, 2], vertical_alignment="center")
        with col[0]:
            st.session_state["forward_default_model_error_as"] = st.toggle(
                label="Forward D. Model Error",
                value=st.session_state["forward_default_model_error_as"],
                key="forward_default_model_error_as_toggle",
            )
        with col[1]:
            st.write("Visualizes the pointwise absolute error between the forward default model L(D)(τ) and F(τ).")

        col = st.columns([1, 2], vertical_alignment="center")
        with col[0]:
            st.session_state["noise_samples_as"] = st.toggle(
                label="Noise Samples",
                value=st.session_state["noise_samples_as"],
                key="noise_samples_as_toggle",
            )
        with col[1]:
            st.write("Visualizes the noise samples.")

    with st.expander("Result Plotting Options", expanded=st.session_state["auto_expand_as"]):
        col = st.columns([1, 2], vertical_alignment="center")
        with col[0]:
            st.session_state["coefficients_as"] = st.toggle(
                label="Coefficients",
                value=st.session_state["coefficients_as"],
                key="coefficients_as_toggle",
            )
        with col[1]:
            st.write("Displays the coefficients c(α) of the model.")

        col = st.columns([1, 2], vertical_alignment="center")
        with col[0]:
            st.session_state["model_as"] = st.toggle(
                label="Model",
                value=st.session_state["model_as"],
                key="model_as_toggle",
            )
        with col[1]:
            st.write("Visualizes the model S(ω) using the provided data for ω.")

        col = st.columns([1, 2])
        with col[0]:
            st.session_state["forward_model_as"] = st.toggle(
                label="Forward Model",
                value=st.session_state["forward_model_as"],
                key="forward_model_as_toggle",
            )
        with col[1]:
            st.write(
                "Visualizes the forward model L(S)(τ) using the provided data for τ."
            )

        col = st.columns([1, 2])
        with col[0]:
            st.session_state["forward_model_error_as"] = st.toggle(
                label="Forward Model Error",
                value=st.session_state["forward_model_error_as"],
                key="forward_model_error_as_toggle",
            )
        with col[1]:
            st.write(
                "Visualizes the pointwise absolute error between the forward model L(S)(τ) and F(τ)."
            )

    col = st.columns([1, 1, 5])
    with col[0]:
        if st.button("Apply", use_container_width=True):
            if save_settings(
                PATH_SETTINGS,
                load_settings(st.session_state),
            ):
                st.toast("Settings saved successfully.", icon="✅")
            else:
                st.toast("Failed to save settings.", icon="❌")
    with col[1]:
        if st.button("Reset", use_container_width=True):
            reset_settings()
            save_settings(PATH_SETTINGS, load_settings(st.session_state))
            st.rerun()


if __name__ == "__main__":
    main()
