import numpy as np
import streamlit as st

from streamlit import plotly_chart, session_state as state
from pylit.frontend.utils import settings_manager
from pylit.backend.core.plot_utils import (
    plot_default_model,
    plot_forward_default_model,
    plot_forward_default_model_error,
    plot_coeffs,
    plot_model,
    plot_forward_model,
    plot_forward_model_error,
)
from pylit.backend import methods, optimize, models
from pylit.backend.core import (
    noise_conv,
    noise_iid,
    Experiment,
)
from pylit.frontend.settings import (
    NOISES_IID,
    NOISES_IID_PARAM_MAP,
    NOISES_CONV,
    NOISES_CONV_PARAM_MAP,
    METHODS,
    METHODS_PARAM_MAP,
    OPTIMIZER,
    OPTIM_PARAM_MAP,
    MODELS,
    MODEL_PARAM_MAP,
)
from pylit.frontend.components import (
    OpenExperiment,
    ExportDataFiles,
    NameParams,
    LatexRow,
    HR,
    Label,
    PlotlyDataSamples,
    DictionaryTable,
)

from pylit.global_settings import MOMENT_ORDERS, COLOR_F


def main():

    def init_experiment(name: str):
        state.exp = Experiment(
            workspace=state.workspace_as,
            name=name,
        )

    # Menu button ids
    experiment_go_back_button_id = "experiment_go_back_button"
    experiment_refresh_button_id = "experiment_refresh_button"

    # Menu button click events (must be here to trigger state updates)
    if experiment_go_back_button_id in state and state[experiment_go_back_button_id]:
        state.clear()

    if experiment_refresh_button_id in state and state[experiment_refresh_button_id]:
        name = state.exp.name
        state.clear()
        settings_manager()
        init_experiment(name)
        # st.rerun()

    settings_manager()

    st.set_page_config(
        page_title="Experiment",
        page_icon="🚀",
        layout="wide" if state["wide_mode_as"] else "centered",
    )

    if "exp" not in state:
        state.exp = None

    # Assign state["exp"] to a local variable for easy access
    if state.exp is None:

        def on_change_name():
            init_experiment(
                name=state.experiment_name,
            )

        # Add custom CSS to reduce margin
        Label(
            text="New Experiment",
            mb=15,
        )

        st.text_input(
            label="Experiment Name",
            label_visibility="collapsed",
            placeholder="Enter a name for the experiment.",
            key="experiment_name",
            on_change=on_change_name,
        )

        st.write("<br/>", unsafe_allow_html=True)

        name = OpenExperiment(
            my_id="open_experiment",
            workspace=state.workspace_as,
        )
        if name is not None:
            init_experiment(name)
            st.rerun()  #  NOTE poor state handling

        # as soon as a name is clicked and it is a valid experiment, give back the name of the experiment -> new component!

    else:

        if not isinstance(state.exp, Experiment):
            st.error("Experiment object is not of type Experiment.")
            return

        col = st.columns([10, 60, 10, 10, 10], vertical_alignment="center")

        with col[0]:
            st.button(
                label="↩",
                key=experiment_go_back_button_id,
                use_container_width=True,
            )

        with col[1]:
            st.markdown(
                f"""
                    <div style='text-align:center;font-size:14pt;'>
                        <strong>{state.exp.name}</strong>
                    </div>
                """,
                unsafe_allow_html=True,
            )

        with col[2]:
            st.button(
                label="🔄",
                key=experiment_refresh_button_id,
                use_container_width=True,
            )

        with col[3]:
            if st.button(
                label="💾",
                use_container_width=True,
            ):
                state.exp.create_run_py(
                    coefficients=state.coefficients_as,
                    model=state.model_as,
                    forward_model=state.forward_model_as,
                    forward_model_error=state.forward_model_error_as,
                )
                state.exp.save_config()
                st.toast(
                    "Saved configuration successfully.",
                    icon="✅",
                )

        with col[4]:
            if st.button(
                label="🚀",
                use_container_width=True,
            ):
                state.exp.create_run_py(
                    coefficients=state.coefficients_as,
                    model=state.model_as,
                    forward_model=state.forward_model_as,
                    forward_model_error=state.forward_model_error_as,
                )
                state.exp.save_config()
                st.toast(
                    "Saved configuration automatically.",
                    icon="✅",
                )
                state.exp.fit_model()

        st.write(
            "<div style='margin-bottom:50px;'></div>",
            unsafe_allow_html=True,
        )

        tab = st.tabs(
            [
                "Data F",
                "Data D",
                "Noise F",
                "Description",
                "Model",
                "Method",
                "Optimizer",
                "Result",
            ],
        )

        css = """
        <style>
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size:14pt;
            }
        </style>
        """

        st.markdown(css, unsafe_allow_html=True)

        with tab[0]:
            if ExportDataFiles(
                my_id="data_csv_column_exporter_F",
                first_var_label="τ",
                second_var_label="F(τ)",
                default_directory=state.workspace_as,
                export_path=state.exp.path_F,
            ):
                # Automatically import the data and prepares it
                state.exp.import_F()

        with tab[1]:
            if not state.exp.imported_F:
                st.warning(
                    "Please export F.csv first.",
                    icon="⚠️",
                )
            else:
                if ExportDataFiles(
                    my_id="data_csv_column_exporter_D",
                    first_var_label="ω",
                    second_var_label="D(ω)",
                    default_directory=state.workspace_as,
                    export_path=state.exp.path_D,
                ):
                    # Automatically import the data and prepares it
                    state.exp.import_D(
                        non_negative=state.non_negative_as,
                        detailed_balance=state.detailed_balance_as,
                    )

        with tab[2]:
            if not state.exp.imported:
                st.warning(
                    "Please export F.csv and D.csv first.",
                    icon="⚠️",
                )
            else:
                # --- Noise --- #
                col1, col2 = st.columns([1, 1])
                with col1:
                    state.exp.config.noiseActive = st.toggle(
                        label="Noise",
                        value=state.exp.config.noiseActive,
                        key=f"noise_active",
                    )

                    if state.exp.config.noiseActive:
                        noiseName = state.exp.config.noiseName
                        noiseName = "" if noiseName is None else noiseName
                        noiseParams = state.exp.config.noiseParams
                        noiseParams = {} if noiseParams is None else noiseParams
                        HR()

                        state.exp.config.noiseParams = NameParams(
                            _selected_ref=state.exp.config.noiseName,
                            my_id=f"noise_iid",
                            options=NOISES_IID,
                            ref=noise_iid,
                            param_map=NOISES_IID_PARAM_MAP.insert_values(noiseParams),
                            name=noiseName,
                        )

                with col2:
                    if state.exp.config.noiseActive:
                        state.exp.config.noiseConvActive = (
                            st.toggle(
                                label="Noise Convolution",
                                value=state.exp.config.noiseConvActive,
                                key=f"noise_conv_active",
                            )
                            and state.exp.config.noiseActive
                        )

                        if state.exp.config.noiseConvActive:
                            noiseConvName = state.exp.config.noiseConvName
                            noiseConvName = (
                                "" if noiseConvName is None else noiseConvName
                            )
                            noiseConvParams = state.exp.config.noiseConvParams
                            noiseConvParams = (
                                {} if noiseConvParams is None else noiseConvParams
                            )
                            HR()

                            state.exp.config.noiseConvParams = NameParams(
                                _selected_ref=state.exp.config.noiseConvName,
                                my_id=f"noise_conv",
                                options=NOISES_CONV,
                                ref=noise_conv,
                                param_map=NOISES_CONV_PARAM_MAP.insert_values(
                                    noiseConvParams
                                ),
                                name=noiseConvName,
                            )

                HR()

                cols = st.columns([1, 1, 2])
                with cols[0]:
                    if st.button(
                        label="Apply",
                        use_container_width=True,
                        key="apply_noise_button",
                    ):
                        state.exp.apply_noise_F()
                        st.toast(
                            "Applied Noise successfully to F.",
                            icon="✅",
                        )

                with cols[1]:
                    if st.button(
                        label="Reset",
                        use_container_width=True,
                        key="reset_noise_button",
                    ):
                        state.exp.reset_noise_F()
                        st.toast(
                            "Reset Noise successfully from F.",
                            icon="✅",
                        )
                # --- --- --- #

        with tab[3]:
            if not state.exp.imported:
                st.warning(
                    "Please export F.csv and D.csv first.",
                    icon="⚠️",
                )
            else:
                # --- Description --- #

                # Characteristics
                Label(
                    text="Characteristics",
                    mb=-5,
                )
                LatexRow(
                    [
                        r"\mathbb{E}_D(\omega) \approx "
                        + str(np.round(state.exp.prep.expD, 2)),
                        r"\mathbb{V}^2_D(\omega) \approx "
                        + str(np.round(state.exp.prep.stdD, 2)),
                        r"\max_\tau \left|\mathcal{L}[D]-F\right| (\tau) \approx "
                        + f"{state.exp.prep.forwardDMaxError:.2e}",
                    ]
                )

                # Frequency Moments
                HR()
                Label(
                    text="Frequency Moments",
                    mb=10,
                )
                DictionaryTable(
                    {
                        "α": ["α = " + str(alpha) for alpha in MOMENT_ORDERS],
                        "⟨ωᵅ⟩": [
                            f"{moment:.2e}" for moment in state.exp.prep.freqMomentsD
                        ],
                    }
                )

                # Plot Default Model
                if state.default_model_as:
                    HR()
                    Label(
                        text="Default Model",
                        mb=-15,
                    )
                    plotly_chart(
                        plot_default_model(state.exp),
                        use_container_width=True,
                    )

                # Plot Forward Default Model
                if state.forward_default_model_as:
                    HR()
                    Label(
                        text="Forward Default Model",
                        mb=-15,
                    )
                    plotly_chart(
                        plot_forward_default_model(state.exp),
                        use_container_width=True,
                    )

                # Plot Forward Default Model Error
                if state.forward_default_model_error_as:
                    HR()
                    Label(
                        text="Forward Default Model Error",
                        mb=-15,
                    )
                    plotly_chart(
                        plot_forward_default_model_error(state.exp),
                        use_container_width=True,
                    )

                # Noise Samples
                if (
                    state.noise_samples_as
                    and state.exp.config.noiseActive
                    and state.exp.prep.noiseF is not None
                    and len(state.exp.prep.noiseF) > 0
                ):
                    HR()
                    Label(
                        text="Noise Samples",
                        mb=-15,
                    )
                    PlotlyDataSamples(state.exp.prep.noiseF, COLOR_F)

                # --- --- --- #

        with tab[4]:
            if not state.exp.imported:
                st.warning(
                    "Please export F.csv and D.csv first.",
                    icon="⚠️",
                )
            else:
                # --- Model --- #
                modelName = state.exp.config.modelName
                modelName = "" if modelName is None else modelName
                modelParams = state.exp.config.modelParams
                modelParams = {} if modelParams is None else modelParams
                state.exp.config.modelParams = NameParams(
                    _selected_ref=state.exp.config.modelName,
                    my_id=f"model",
                    options=MODELS,
                    ref=models,
                    param_map=MODEL_PARAM_MAP(state.exp).insert_values(modelParams),
                    name=modelName,
                )

        with tab[5]:
            if not state.exp.imported:
                st.warning(
                    "Please export F.csv and D.csv first.",
                    icon="⚠️",
                )
            else:
                # --- Method --- #
                methodName = state.exp.config.methodName
                methodName = "" if methodName is None else methodName
                methodParams = state.exp.config.methodParams
                methodParams = {} if methodParams is None else methodParams
                state.exp.config.methodParams = NameParams(
                    _selected_ref=state.exp.config.methodName,
                    my_id=f"method",
                    options=METHODS,
                    ref=methods,
                    param_map=METHODS_PARAM_MAP(state.exp).insert_values(methodParams),
                    name=methodName,
                )
                # --- --- --- #

        with tab[6]:
            if not state.exp.imported:
                st.warning(
                    "Please export F.csv and D.csv first.",
                    icon="⚠️",
                )
            else:
                # --- Optimizer --- #
                optimName = state.exp.config.optimName
                optimName = "" if optimName is None else optimName
                optimParams = state.exp.config.optimParams
                optimParams = {} if optimParams is None else optimParams
                state.exp.config.optimParams = NameParams(
                    _selected_ref=state.exp.config.optimName,
                    my_id=f"optim",
                    options=OPTIMIZER,
                    ref=optimize,
                    param_map=OPTIM_PARAM_MAP.insert_values(optimParams),
                    name=optimName,
                )

                cols = st.columns([1, 2])
                with cols[0]:
                    state.exp.config.x0Reset = st.toggle(
                        "Reset x0",
                        key="x0_reset",
                        value=state.exp.config.x0Reset,
                    )
                with cols[1]:
                    st.markdown(
                        "Resets the initial guess to the default value.",
                    )

                cols = st.columns([1, 2])
                with cols[0]:
                    state.exp.config.adaptiveActive = st.toggle(
                        "Adaptive",
                        key="adaptive_active",
                        value=state.exp.config.adaptiveActive,
                    )
                with cols[1]:
                    st.markdown(
                        "Activates adaptive optimization mode, where the algorithm incrementally decreases the kernel widths in the model space until further reductions in error are no longer achieved.",
                    )

                cols = st.columns([1, 2])
                with cols[0]:
                    state.exp.config.adaptiveResiduumMode = st.toggle(
                        "Residuum Mode",
                        key="adaptive_residuum_mode",
                        value=state.exp.config.adaptiveResiduumMode,
                    )
                with cols[1]:
                    st.markdown(
                        "If enabled, the adaptive optimization mode will be based on the residuum error instead of the chosen method error.",
                    )
                # --- --- --- #

        with tab[7]:
            if state.exp.output is not None:
                # Characteristics
                Label(
                    text="Characteristics",
                    mb=-5,
                )
                LatexRow(
                    [
                        r"\text{avg}\ \mathbb{E}_S(\omega) \approx "
                        + str(np.round(np.mean(state.exp.output.expS), 2)),
                        r"\text{avg}\ \mathbb{V}^2_S(\omega) \approx "
                        + str(np.round(np.mean(state.exp.output.stdS), 2)),
                        r"\text{avg}\ \max_\tau \left|\mathcal{L}[S] - F\right|(\tau) \approx "
                        + f"{np.mean(state.exp.output.forwardSMaxError):.2e}",
                    ]
                )

                # Frequency Moments
                HR()
                Label(
                    text="Frequency Moments",
                    mb=10,
                )
                DictionaryTable(
                    {
                        "α": ["α = " + str(alpha) for alpha in MOMENT_ORDERS],
                        "avg ⟨ωᵅ⟩": [
                            f"{np.mean(moment):.2e}"
                            for moment in state.exp.output.freqMomentsS.T
                        ],
                    }
                )

                # Coefficients
                if state.coefficients_as:
                    HR()
                    Label(
                        text="Coefficients",
                        mb=-15,
                    )
                    plotly_chart(
                        plot_coeffs(state.exp),
                        use_container_width=True,
                    )

                # Plot Model
                if state.model_as:
                    HR()
                    Label(
                        text="Model",
                        mb=-15,
                    )
                    plotly_chart(
                        plot_model(state.exp),
                        use_container_width=True,
                    )

                # Plot Forward Model
                if state.forward_model_as:
                    HR()
                    Label(
                        text="Forward Model",
                        mb=-15,
                    )
                    plotly_chart(
                        plot_forward_model(state.exp),
                        use_container_width=True,
                    )

                # Plot Forward Model Error
                if state.forward_model_error_as:
                    HR()
                    Label(
                        text="Forward Model Error",
                        mb=-15,
                    )
                    plotly_chart(
                        plot_forward_model_error(state.exp),
                        use_container_width=True,
                    )

            else:  # TODO Test this!
                st.warning(
                    "There is nothing to display yet.",
                    icon="⚠️",
                )
                st.info(
                    "- Save your configurations by pressing 💾\n"
                    "- Either press 🚀 or run via the terminal\n"
                )

                st.info(
                    "To run via terminal, use the following commands.\n\n"
                    "```bash\n"
                    "conda activate pylit\n"
                    f"python {state.exp.path_run}\n"
                    "conda deactivate\n"
                    "```"
                )


if __name__ == "__main__":
    main()
