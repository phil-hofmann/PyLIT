import numpy as np
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from pylit.frontend.utils import settings_manager
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
    SCALINGS,
    SCALINGS_PARAM_MAP,
    MODELS,
    MODEL_PARAM_MAP,
)
from pylit.global_settings import ARRAY
from pylit.frontend.components import ExportDataFiles, NameParams, DisplayFigure
from pylit.frontend.components import OpenExperiment
from pylit.frontend.core import Param, ParamMap


def main():
    settings_manager()

    st.set_page_config(
        page_title="Experiment",
        page_icon="🚀",
        layout="wide" if st.session_state["wide_mode"] else "centered",
    )

    if "exp" not in st.session_state:
        st.session_state["exp"] = None
    if "preprocessed" not in st.session_state:
        st.session_state["preprocessed"] = False

    def init_experiment(name: str):
        st.session_state["exp"] = Experiment(
            workspace=st.session_state["workspace"],
            name=name,
            make=True,
        )

    # Assign st.session_state["exp"] to a local variable for easy access
    if not st.session_state["exp"] or st.session_state["exp"] is None:

        def on_change_name():
            init_experiment(
                name=st.session_state["experiment_name"],
            )

        st.text_input(
            "Experiment Name",
            placeholder="Enter a name for the experiment.",
            key="experiment_name",
            on_change=on_change_name,
        )

        st.write("<br/>", unsafe_allow_html=True)

        name = OpenExperiment(
            my_id="open_experiment",
            workspace=st.session_state["workspace"],
        )
        if name is not None:
            init_experiment(name)
            st.rerun()  #  NOTE poor state handling

        # as soon as a name is clicked and it is a valid experiment, give back the name of the experiment -> new component!

    else:
        exp = st.session_state["exp"]

        if not isinstance(exp, Experiment):
            st.error("Experiment object is not of type Experiment.")
            return

        col1, col2, col3 = st.columns([95, 5, 5])

        with col1:
            st.write(
                f"""#### Name: {exp.name}""",
                unsafe_allow_html=True,
            )
        with col2:
            if st.button("🔄"):
                name = exp.name
                st.session_state["exp"] = None
                init_experiment(name)
                st.rerun()
        with col3:
            if st.button("❌"):
                # Page reload TODO: solve this in a better way
                streamlit_js_eval(js_expressions="parent.window.location.reload()")

        st.write(
            "<hr style='margin:0;padding:0;padding-top:0px;'/>",
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            [
                "🛢️ Export F",
                "🛢️ Export S",
                "🎛️ &nbsp; Data Adjustments",
                "🎛️ &nbsp; Model, Optimizer and Method",
                "📊 &nbsp; Output",
                "📄 &nbsp; Summary",
                "📝 &nbsp; Results",
            ]
        )

        st.markdown(
            "<hr style='margin:0;padding:0;padding-top:5px;'/><br/>",
            unsafe_allow_html=True,
        )

        with tab1:
            if ExportDataFiles(
                my_id="data_csv_column_exporter_F",
                default_directory=st.session_state["workspace"],
                export_path=exp.directory + "/F.csv",
            ):
                exp.import_F()

        with tab2:
            if ExportDataFiles(
                my_id="data_csv_column_exporter_S",
                default_directory=st.session_state["workspace"],
                export_path=exp.directory + "/S.csv",
            ):
                exp.import_S()

        with tab3:
            st.markdown(
                "<br/><b>Scale and Adjust</b><hr style='margin:0;padding:0'/>",
                unsafe_allow_html=True,
            )
            if not exp.exported:
                st.warning(
                    "Please export F.csv and S.csv first.",
                    icon="⚠️",
                )
            else:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    exp.config.scaleMaxF = st.toggle(
                        label="Scale F (max-entry)",
                        value=exp.config.scaleMaxF,
                        key=f"scale_max_F",
                    )

                with col2:
                    exp.config.PosS = st.toggle(
                        label="Positive Part of S",
                        value=exp.config.PosS,
                        key=f"pos_S",
                    )

                with col3:
                    exp.config.ExtS = st.toggle(
                        label="Extend S to negative axis",
                        value=exp.config.ExtS,
                        key=f"ext_S",
                    )

                with col4:
                    exp.config.trapzS = st.toggle(
                        label="Scale S (trapezoidal rule)",
                        value=exp.config.trapzS,
                        key=f"trapz_S",
                    )

                col1, col2 = st.columns([1, 1])

                with col1:
                    exp.config.noiseActive = st.toggle(
                        label="Noise",
                        value=exp.config.noiseActive,
                        key=f"noise_active",
                    )

                    if exp.config.noiseActive:
                        noiseName = exp.config.noiseName
                        noiseName = "" if noiseName is None else noiseName
                        noiseParams = exp.config.noiseParams
                        noiseParams = {} if noiseParams is None else noiseParams
                        st.markdown(
                            "<hr style='margin:0;padding:0'/>",
                            unsafe_allow_html=True,
                        )

                        (
                            exp.config.noiseName,
                            exp.config.noiseParams,
                        ) = NameParams(
                            my_id=f"noise_iid",
                            options=NOISES_IID,
                            ref=noise_iid,
                            param_map=NOISES_IID_PARAM_MAP.insert_values(noiseParams),
                            name=noiseName,
                        )

                with col2:
                    if exp.config.noiseActive:
                        exp.config.noiseConvActive = (
                            st.toggle(
                                label="Noise Convolution",
                                value=exp.config.noiseConvActive,
                                key=f"noise_conv_active",
                            )
                            and exp.config.noiseActive
                        )

                        if exp.config.noiseConvActive:
                            noiseConvName = exp.config.noiseConvName
                            noiseConvName = (
                                "" if noiseConvName is None else noiseConvName
                            )
                            noiseConvParams = exp.config.noiseConvParams
                            noiseConvParams = (
                                {} if noiseConvParams is None else noiseConvParams
                            )
                            st.markdown(
                                "<hr style='margin:0;padding:0'/>",
                                unsafe_allow_html=True,
                            )

                            (
                                exp.config.noiseConvName,
                                exp.config.noiseConvParams,
                            ) = NameParams(
                                my_id=f"noise_conv",
                                options=NOISES_CONV,
                                ref=noise_conv,
                                param_map=NOISES_CONV_PARAM_MAP.insert_values(
                                    noiseConvParams
                                ),
                                name=noiseConvName,
                            )

                st.markdown("<hr style='margin:0;padding:0'/>", unsafe_allow_html=True)

                if st.button("Prepare data", key="prepro_btn"):
                    with st.spinner("Preparing data ..."):
                        exp.prepare()
                        # st.session_state["preprocessed"] = True
                st.markdown("<hr style='margin:0;padding:0'/>", unsafe_allow_html=True)

                if exp.prepared:
                    st.table(
                        [
                            {
                                "Var": "𝜏",
                                "Min": exp.prep.tauMin,
                                "Max": exp.prep.tauMax,
                            },
                            {
                                "Var": "ω",
                                "Min": exp.prep.modifiedOmegaMin,
                                "Max": exp.prep.modifiedOmegaMax,
                            },
                        ]
                    )

                    st.table(
                        [
                            {
                                "Expected Value S": exp.prep.expS,
                                "Standard deviation S": exp.prep.stdS,
                                "Maximal Forward Error S": exp.prep.forwardModifiedSMaxError,
                            }
                        ]
                    )

                    fig_data, fig_forward = exp.plot_prep()
                    DisplayFigure(fig_data)
                    DisplayFigure(fig_forward)

        with tab4:
            if not exp.prepared:
                st.warning(
                    "Please prepare the data first.",
                    icon="⚠️",
                )
            else:
                # Create columns
                col1, col2 = st.columns([1, 1])

                # --- Method --- #
                with col1:
                    st.markdown(
                        f"""<br/><b>Method</b><hr style='margin:0;padding:0'/>""",
                        unsafe_allow_html=True,
                    )
                    methodName = exp.config.methodName
                    methodName = "" if methodName is None else methodName
                    methodParams = exp.config.methodParams
                    methodParams = {} if methodParams is None else methodParams
                    # st.write(methodParams) TODO Remove
                    (
                        exp.config.methodName,
                        exp.config.methodParams,
                    ) = NameParams(
                        my_id=f"method",
                        options=METHODS,
                        ref=methods,
                        param_map=METHODS_PARAM_MAP(exp).insert_values(methodParams),
                        name=methodName,
                    )
                # --- --- --- #

                # --- Optimizer --- #
                with col2:
                    st.markdown(
                        "<br/><b>Optimizer</b><hr style='margin:0;padding:0'/>",
                        unsafe_allow_html=True,
                    )
                    optimName = exp.config.optimName
                    optimName = "" if optimName is None else optimName
                    optimParams = exp.config.optimParams
                    optimParams = {} if optimParams is None else optimParams
                    (
                        exp.config.optimName,
                        exp.config.optimParams,
                    ) = NameParams(
                        my_id=f"optim",
                        options=OPTIMIZER,
                        ref=optimize,
                        param_map=OPTIM_PARAM_MAP.insert_values(optimParams),
                        name=optimName,
                    )
                    exp.config.x0Reset = st.toggle(
                        "Reset x0",
                        key="x0_reset",
                        value=exp.config.x0Reset,
                    )
                    exp.config.adaptiveActive = st.toggle(
                        "Adaptive",
                        key="adaptive_active",
                        value=exp.config.adaptiveActive,
                    )
                    exp.config.adaptiveResiduumMode = st.toggle(
                        "Residuum Mode",
                        key="adaptive_residuum_mode",
                        value=exp.config.adaptiveResiduumMode,
                    )
                # --- --- --- #

                # --- Model --- #
                st.markdown(
                    f"""<br/><b>Model</b><hr style='margin:0;padding:0'/>""",
                    unsafe_allow_html=True,
                )
                scalingName = exp.config.scalingName
                scalingName = "" if scalingName is None else scalingName
                scalingParams = exp.config.scalingParams
                scalingParams = {} if scalingParams is None else scalingParams
                (
                    exp.config.scalingName,
                    exp.config.scalingParams,
                ) = NameParams(
                    my_id=f"scaling",
                    options=SCALINGS,
                    ref=models.scaling,
                    param_map=SCALINGS_PARAM_MAP.insert_values(scalingParams),
                    label="Scaling",
                    name=scalingName,
                )

                modelName = exp.config.modelName
                modelName = "" if modelName is None else modelName
                modelParams = exp.config.modelParams
                modelParams = {} if modelParams is None else modelParams
                (
                    exp.config.modelName,
                    exp.config.modelParams,
                ) = NameParams(
                    my_id=f"model",
                    options=MODELS,
                    ref=models,
                    param_map=MODEL_PARAM_MAP(exp).insert_values(modelParams),
                    name=modelName,
                )

        with tab5:
            if exp.ready_to_finish:
                st.markdown(
                    f"""<br/><b>Plotting Options</b><hr style='margin:0;padding:0'/>""",
                    unsafe_allow_html=True,
                )

                exp.config.plot_coeffs = st.toggle(
                    "Model coefficients",
                    key="plot_coeffs",
                    value=exp.config.plot_coeffs,
                )

                exp.config.plot_model = st.toggle(
                    "Model",
                    key="plot_model",
                    value=exp.config.plot_model,
                )

                exp.config.plot_forward_model = st.toggle(
                    "Forwarded model",
                    key="plot_forward_model",
                    value=exp.config.plot_forward_model,
                )

                exp.config.plot_error_model = st.toggle(
                    "Model error",
                    key="plot_error_model",
                    value=exp.config.plot_error_model,
                )

                exp.config.plot_error_forward_model = st.toggle(
                    "Forwarded model error",
                    key="plot_error_forward_model",
                    value=exp.config.plot_error_forward_model,
                )

                st.write(
                    "<hr style='margin:0;padding:0;padding-top:5px;'/>",
                    unsafe_allow_html=True,
                )

                if st.button("Finish Experiment Setup"):
                    if exp.create_run():  # TODO create this function
                        st.success("✅ &nbsp; Experiment setup completed.")
                    else:
                        # TODO is this case necessary?
                        st.warning(
                            "Some error message.",
                            icon="⚠️",
                        )
            else:
                st.warning(
                    "Please finish the 'Data Adjustment' and 'Model, Optimizer and Method' steps first.",
                    icon="⚠️",
                )

        with tab6:
            st.write(exp.config)
            st.write(exp.prep)

        with tab7:
            displayed_any = False
            if exp.output is not None:
                # TODO add these checks also in the experiment class methods!
                if exp.output.coefficients is not None and exp.config.plot_coeffs:
                    DisplayFigure(exp.plot_coeffs())
                    displayed_any = True
                if (
                    exp.prep.modifiedS is not None
                    and exp.output.valsS is not None
                    and exp.config.plot_model
                ):
                    DisplayFigure(exp.plot_model())
                    displayed_any = True
                if (
                    exp.prep.modifiedF is not None
                    and exp.output.valsF is not None
                    and exp.config.plot_forward_model
                ):
                    DisplayFigure(exp.plot_forward_model())
                    displayed_any = True
                if (
                    exp.prep.modifiedS is not None
                    and exp.output.valsS is not None
                    and exp.config.plot_error_model
                ):
                    DisplayFigure(exp.plot_error_model())
                    displayed_any = True
                if (
                    exp.prep.modifiedF is not None
                    and exp.output.valsF is not None
                    and exp.config.plot_error_forward_model
                ):
                    DisplayFigure(exp.plot_error_forward_model())
                    displayed_any = True

            if not displayed_any:
                st.warning(
                    "There is nothing to display yet.",
                    icon="⚠️",
                )


if __name__ == "__main__":
    main()
