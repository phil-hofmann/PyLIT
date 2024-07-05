import numpy as np
import streamlit as st
from dataclasses import asdict  # TODO Do I really need that?!?
from pylit import ui
from pylit.core import noise_conv, noise_iid  # TODO Do I really need that?!?
from pylit.core.experiment import Experiment
from pylit.global_settings import ARRAY
from pylit.ui.molecules.export_data_files import export_data_files


def main():
    st.set_page_config(
        page_title="Experiment",
        page_icon="🚀",
        layout="wide" if st.session_state["wide_mode"] else "centered",
    )

    ui.components.settings_manager()
    ui.components.file_menu()

    if "exp" not in st.session_state:
        st.session_state["exp"] = None
    if "preprocessed" not in st.session_state:
        st.session_state["preprocessed"] = False


    # Assign st.session_state["exp"] to a local variable for easier access
    if not st.session_state["exp"]:
        def on_change_name():
            st.session_state["exp"] = Experiment(
                name=st.session_state["experiment_name"],
                make=True,
            )
            
        st.text_input(
            "Experiment Name",
            placeholder="Enter a name for the experiment.",
            key="experiment_name",
            on_change=on_change_name,
        )
    else:
        exp = st.session_state["exp"]

        if not isinstance(exp, Experiment):
            st.error("Experiment object is not of type Experiment.")
            return

        st.write(
            f"""
                #### Name: {exp.name}
                <hr style='margin:0;padding:0;padding-top:5px;'/>
                """,
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "🛢️ Export F",
                "🛢️ Export S",
                "📊 &nbsp; Data Adjustments",
                "🎛️ &nbsp; Model, Optimizer and Method",
                "🚀 &nbsp; Export Experiment",
            ]
        )

        st.markdown(
            "<hr style='margin:0;padding:0;padding-top:5px;'/><br/>",
            unsafe_allow_html=True,
        )

        with tab1:
            if export_data_files(
                my_id="data_csv_column_exporter_F",
                default_directory=st.session_state["workspace"],
                export_path=exp.name + "/F.csv",
            ):
                exp.import_F()
            
        with tab2:
            if export_data_files(
                my_id="data_csv_column_exporter_S",
                default_directory=st.session_state["workspace"],
                export_path=exp.name + "/S.csv",
            ):
                exp.import_S()

        with tab3:
            st.markdown(
                "<br/><b>Scale and Adjust</b><hr style='margin:0;padding:0'/>",
                unsafe_allow_html=True,
            )

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
                    st.markdown(
                        "<hr style='margin:0;padding:0'/>",
                        unsafe_allow_html=True,
                    )

                    (
                        exp.config.noiseName,
                        exp.config.noiseParams,
                    ) = ui.components.name_params(
                        my_id=f"noise_iid",
                        idx=i,
                        options=ui.settings.NOISES_IID,
                        ref=noise_iid,
                        param_map=ui.settings.NOISES_IID_PARAM_MAP,
                    )

            with col2:
                if exp.config.noiseActive:
                    exp.config.noiseConvActive = (
                        st.toggle(
                            label="Noise Convolution",
                            value=exp.config.noiseConvActive,
                            key=f"noise_convolution_{i}",
                        )
                        and exp.config.noiseActive
                    )

                    if exp.config.noiseConvActive:
                        st.markdown(
                            "<hr style='margin:0;padding:0'/>",
                            unsafe_allow_html=True,
                        )

                        (
                            exp.config.noiseConvName,
                            exp.config.noiseConvParams,
                        ) = ui.components.name_params(
                            key=f"noise_conv",
                            idx=i,
                            options=ui.settings.NOISES_CONV,
                            ref=noise_conv,
                            param_map=ui.settings.NOISES_CONV_PARAM_MAP,
                        )

            st.markdown("<hr style='margin:0;padding:0'/>", unsafe_allow_html=True)

            if st.button("Prepare data", key="prepro_btn"):
                with st.spinner("Preparing data ..."):
                    exp.prepare()
                    st.session_state["preprocessed"] = True
            st.markdown("<hr style='margin:0;padding:0'/>", unsafe_allow_html=True)

            if st.session_state["preprocessed"]:
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
                        }
                    ]
                )

            # if "fig_data" in st.session_state:
            #     fig_data = st.session_state["fig_data"]
            #     ui.components.display_figure(fig_data)

        with tab4:
            # Create tabs for each data name
            tabs = st.tabs(
                [
                    f"**{data_name}**"
                    for data_name in st.session_state["config"]["data"]["names"]
                ]
            )

            for i in range(N):
                with tabs[i]:
                    # Create columns
                    col11, col12 = st.columns([1, 1])

                    with col11:
                        # Method

                        st.markdown(
                            f"""<br/><b>Method</b><hr style='margin:0;padding:0'/>""",
                            unsafe_allow_html=True,
                        )
                        ui.components.name_params(
                            key=f"method",
                            idx=i,
                            options=ui.settings.METHODS,
                            ref=pylit.methods,
                            param_map=ui.settings.METHODS_PARAM_MAP,
                        )

                    with col12:
                        # --- Optimizer --- # TODO CHANGE TO name_params with param_map ignore attributes
                        st.markdown(
                            "<br/><b>Optimizer</b><hr style='margin:0;padding:0'/>",
                            unsafe_allow_html=True,
                        )
                        st.selectbox(
                            label=f"Name",
                            options=ui.settings.OPTIMIZER(name=True),
                            key=f"optim_name_{i}",
                            index=ui.settings.OPTIMIZER.index_of(
                                st.session_state["config"]["optim"]["name"][i]
                            ),
                        )
                        st.session_state["config"]["optim"]["name"][i] = (
                            ui.settings.OPTIMIZER.find(
                                st.session_state[f"optim_name_{i}"]
                            ).ref
                        )

                        # Methodless optimizer attributes

                        for (
                            opt_name,
                            opt_param,
                        ) in ui.settings.OPTIM_PARAM_MAP.items():
                            opt_key = f"ml{i}_{opt_name}"
                            st.session_state["config"]["optim"][opt_name][i] = (
                                ui.components.input(
                                    param_name=opt_param.label,
                                    param_type=opt_param.type,
                                    param_default=opt_param.default,
                                    id=opt_key,
                                )
                            )
                        # --- --- --- #

                    # Model
                    st.markdown(
                        f"""<br/><b>Model</b><hr style='margin:0;padding:0'/>""",
                        unsafe_allow_html=True,
                    )

                    ui.components.name_params(
                        key=f"scaling",
                        idx=i,
                        options=ui.settings.SCALINGS,
                        ref=pylit.models.scaling,
                        param_map=ui.settings.SCALINGS_PARAM_MAP,
                        label="Scaling",
                    )
                    ui.components.name_params(
                        key=f"model",
                        idx=i,
                        options=ui.settings.MODELS,
                        ref=pylit.models,
                        param_map=(
                            ui.param_map.ParamMap(
                                [
                                    ui.param_map.Param(
                                        name="omegas",
                                        type=ARRAY,
                                        default=[
                                            np.round(
                                                st.session_state["exp"].contents[
                                                    "output"
                                                ]["omega_min"],
                                                2,
                                            ),
                                            np.round(
                                                st.session_state["exp"].contents[
                                                    "output"
                                                ]["omega_max"],
                                                2,
                                            ),
                                            int(
                                                len(
                                                    st.session_state["exp"].contents[
                                                        "output"
                                                    ]["S"]["omega"]["contents"]
                                                )
                                                / 5
                                            ),
                                        ],
                                        min_value=-2
                                        * abs(
                                            st.session_state["exp"].contents["output"][
                                                "omega_min"
                                            ]
                                        ),
                                        max_value=2
                                        * abs(
                                            st.session_state["exp"].contents["output"][
                                                "omega_max"
                                            ]
                                        ),
                                    ),
                                    ui.param_map.Param(
                                        name="sigmas",
                                        type=ARRAY,
                                        default=[
                                            np.round(
                                                st.session_state["exp"].contents[
                                                    "output"
                                                ]["S"][name]["std"],
                                                2,
                                            ),
                                            np.round(
                                                10
                                                * st.session_state["exp"].contents[
                                                    "output"
                                                ]["S"][name]["std"],
                                                2,
                                            ),
                                            int(
                                                1
                                                / st.session_state["exp"].contents[
                                                    "output"
                                                ]["S"][name]["std"]
                                            ),
                                        ],
                                        min_value=1e-4,
                                        max_value=20
                                        * st.session_state["exp"].contents["output"][
                                            "S"
                                        ][name]["std"],
                                    ),
                                    ui.param_map.Param(
                                        name="beta",
                                        default=1.0,
                                        ignore=True,
                                    ),
                                    ui.param_map.Param(
                                        name="order",
                                        default="0,1",
                                        ignore=True,
                                    ),
                                ]
                            )
                            if st.session_state["exp"] is not None
                            else None
                        ),
                    )

            # TODO: Change experiment s.t. the following projections are not necessary anymore
            st.session_state["config"]["method"]["name"] = [
                val["name"] for val in st.session_state["method"].values()
            ]
            st.session_state["config"]["method"]["params"] = [
                val["params"] for val in st.session_state["method"].values()
            ]

            scaling = list(st.session_state["scaling"].values())
            model = list(st.session_state["model"].values())

            st.session_state["models"] = [
                {
                    "name": name,
                    "scaling": scaling[i],
                    "model": model[i],
                }
                for i, name in enumerate(st.session_state["config"]["data"]["names"])
            ]

            st.session_state["get_models"] = lambda exp, models=st.session_state[
                "models"
            ]: models  # TODO: exp features?

        # with tab5:
        #     # if st.session_state["nav"] == "JSON":
        #     if st.session_state["view_config_json"]:
        #         with st.expander("Config JSON", expanded=False):
        #             st.write(st.session_state["config"])
        #     if st.session_state["view_models_json"]:
        #         with st.expander("Models JSON", expanded=False):
        #             st.write(st.session_state["models"])
        #     if (
        #         not st.session_state["view_config_json"]
        #         and not st.session_state["view_models_json"]
        #     ):
        #         st.warning("Activate a JSON option in the settings.", icon="⚠️")

        with tab5:
            if st.button("Fit model"):
                with st.spinner("Fitting model ..."):
                    # TODO !
                    if st.session_state["preprocessed"]:
                        # Set config again
                        st.session_state["exp"].config = st.session_state["config"]
                        # TODO
                        (
                            st.session_state["fig_coeffs"],
                            st.session_state["fig_results"],
                        ) = st.session_state["exp"].fit_model(
                            st.session_state["get_models"],
                            store=st.session_state["store"],
                            docx=st.session_state["docx"],
                            giveback=True,
                            interactive=st.session_state["interactive"],
                        )
                        st.session_state["report"] = st.session_state["exp"].contents[
                            "report"
                        ]
                    else:
                        st.warning(
                            "Data not prepared yet. Please go to the Preprocessing tab and preprocess the data first.",
                            icon="⚠️",
                        )
            st.markdown("<hr style='margin:0;padding:0'/>", unsafe_allow_html=True)
            if st.session_state["view_report"]:
                with st.expander("Report", expanded=True):
                    if "report" in st.session_state:
                        st.write(st.session_state["report"])
            if st.session_state["view_coeffs_plot"]:
                with st.expander("Coefficients Plot", expanded=True):
                    if "fig_coeffs" in st.session_state:
                        fig_coeffs = st.session_state["fig_coeffs"]
                        ui.components.display_figure(fig_coeffs)
            with st.expander("Results Plot", expanded=True):
                if "fig_results" in st.session_state:
                    fig_results = st.session_state["fig_results"]
                    ui.components.display_figure(fig_results)


if __name__ == "__main__":
    main()
