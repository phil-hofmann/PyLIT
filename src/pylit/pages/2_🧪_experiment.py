import pylit
import numpy as np
import streamlit as st

from pylit.global_settings import ARRAY

def main():
    pylit.ui.components.settings_manager()

    # Page settings
    st.set_page_config(
        page_title="Experiment",
        page_icon="🚀",
        layout="wide" if st.session_state["wide_mode"] else "centered",
    )

    # Hyperparameters
    N = 3
    
    if "config" not in st.session_state:
        # TODO Put Dummy into settings
        st.session_state["config"] = {
            "name": "",
            "data": {
                "names": [""] * N,
                "noise_active": [False] * N,
                "noise": [{}] * N,  # in percent
                "noise_conv_active": [False] * N,
                "noise_conv": [{}] * N,
                "noise_conv_active": [False] * N,
                "scale": [{
                    "max_F": True,
                    "pos_S": True,
                    "ext_S": True,
                    "trapz_S": True,
                }] * N,
                "file_path_F": "",
                "file_path_S": "",
            },
            "method": {
                "name": [""] * N,
                "params": [{}] * N,
            },
            "optim": {
                "name": [""] * N,
                "maxiter": [pylit.ui.settings.OPTIM_PARAM_MAP["maxiter"].default] * N,
                "tol": [pylit.ui.settings.OPTIM_PARAM_MAP["tol"].default] * N, # TODO change METHODLESS_OPTIMIZER_ATTRS to dictionary
            },
        }

    if "exp" not in st.session_state:
        st.session_state["exp"] = None

    if "preprocessed" not in st.session_state:
        st.session_state["preprocessed"] = False

    pylit.ui.components.file_menu()

    if "name_entered" not in st.session_state:
        st.session_state["name_entered"] = False

    if not st.session_state["name_entered"]:
        def on_change_name():
            st.session_state["name_entered"] = True
            st.session_state["config"]["name"] = st.session_state["experiment_name"]
    
        st.text_input(
            "Experiment Name",
            placeholder="Enter a name for the experiment.",
            key="experiment_name",
            on_change=on_change_name)

    if st.session_state["config"]["name"] != "":
        st.write(f"#### Experiment: {st.session_state['config']['name']} <hr style='margin:0;padding:0;padding-top:5px;'/>", unsafe_allow_html=True)
        
        # Create tabs 
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🛢️ Select F", "🛢️ Select S", "📊  Adjust", "🎛️ Configs", "📝 JSON", "🚀 Fitting"])
        # icons=["database", "database", "clipboard-data", "sliders2", "filetype-json", "rocket"],
        
        st.markdown("<hr style='margin:0;padding:0;padding-top:5px;'/><br/>", unsafe_allow_html=True)

        with tab1:
            st.session_state["config"]["data"]["file_path_F"] = pylit.ui.components.file_selector(id="fs_F")

        with tab2:
            st.session_state["config"]["data"]["file_path_S"] = pylit.ui.components.file_selector(id="fs_S")

        with tab3:
            st.session_state["config"]["data"]["names"] = st.multiselect("Data Names", options=pylit.ui.settings.MODEL_NAMES, default=pylit.ui.settings.MODEL_NAMES)
            st.markdown("<br/>", unsafe_allow_html=True)

            # Create tabs for each data name
            tabs = st.tabs([ f"**{data_name}**" for data_name in st.session_state["config"]["data"]["names"]])

            for i in range(N): 
                with tabs[i]:

                    st.markdown("<br/><b>Scale and Adjust</b><hr style='margin:0;padding:0'/>", unsafe_allow_html=True)
                    # Create columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.toggle(
                                label="Scale F (max-entry)",
                                value=True,
                                key=f"scale_{i}_1",
                            )
                        
                    with col2:
                        st.toggle(
                                label="Positive Part of S",
                                value=True,
                                key=f"scale_{i}_2",
                            )
                        
                    with col3:
                        st.toggle(
                                label="Extend S to negative axis",
                                value=True,
                                key=f"scale_{i}_3",
                            )
                       
                    with col4:
                        st.toggle(
                                label="Scale S (trapezoidal)",
                                value=True,
                                key=f"scale_{i}_4",
                            )
                    
                    st.session_state["config"]["data"]["scale"][i] = {
                        "max_F": st.session_state[f"scale_{i}_1"],
                        "pos_S": st.session_state[f"scale_{i}_2"],
                        "ext_S": st.session_state[f"scale_{i}_3"],
                        "trapz_S": st.session_state[f"scale_{i}_4"],
                    } 

                    st.markdown("<br/>", unsafe_allow_html=True)
                    
                    # Create columns
                    col11, col12 = st.columns([1,1])

                    with col11:
                        st.session_state["config"]["data"]["noise_active"][i] = st.toggle(
                            label="Noise",
                            value=False,
                            key=f"noise_{i}",
                        )

                        if st.session_state["config"]["data"]["noise_active"][i]:
                            st.markdown("<hr style='margin:0;padding:0'/>", unsafe_allow_html=True)
                            
                            # --- Noise IID --- #
                            pylit.ui.components.name_params(
                                key=f"noises_iid",
                                idx=i,
                                options=pylit.ui.settings.NOISES_IID,
                                ref=pylit.noise_iid,
                                param_map=pylit.ui.settings.NOISES_IID_PARAM_MAP,
                            )
                    
                    with col12:
                        if st.session_state["config"]["data"]["noise_active"][i]:

                            st.session_state["config"]["data"]["noise_conv_active"][i] =  st.toggle(
                                label="Noise Convolution",
                                value=False,
                                key=f"noise_convolution_{i}",
                            ) and st.session_state["config"]["data"]["noise_active"][i]

                            if st.session_state["config"]["data"]["noise_conv_active"][i]:
                                st.markdown("<hr style='margin:0;padding:0'/>", unsafe_allow_html=True)
                                
                                # --- Noise Convolution --- #
                                pylit.ui.components.name_params(
                                    key=f"noises_conv",
                                    idx=i,
                                    options=pylit.ui.settings.NOISES_CONV,
                                    ref=pylit.noise_conv,
                                    param_map=pylit.ui.settings.NOISES_CONV_PARAM_MAP,
                                )


            st.markdown("<hr style='margin:0;padding:0'/>", unsafe_allow_html=True)

            if st.button("Preprocess data", key="prepro_btn"):
                with st.spinner('Preprocessing data ...'):
                    if "noises_iid" in st.session_state:
                        st.session_state["config"]["data"]["noise"] = list(st.session_state[f"noises_iid"].values())
                    if "noises_conv" in st.session_state:
                        st.session_state["config"]["data"]["noise_conv"] = list(st.session_state[f"noises_conv"].values())
                    st.session_state["exp"]=pylit.experiments.ExperimentRLRM(path=st.session_state["workspace"], config=st.session_state["config"])
                    st.session_state["fig_data"] = st.session_state["exp"].preprocessing(
                        store=st.session_state["store"],
                        force=st.session_state["force"], 
                        giveback=True,
                        interactive=st.session_state["interactive"]
                    )
                    st.session_state["preprocessed"] = True
            st.markdown("<hr style='margin:0;padding:0'/>", unsafe_allow_html=True)
            
            if st.session_state["preprocessed"]:
                st.session_state["table_data_min_max"] = [
                    {
                        'Var': '𝜏',
                        'Min': st.session_state["exp"].contents["output"]["tau_min"],
                        'Max': st.session_state["exp"].contents["output"]["tau_max"],
                    },
                    {
                        'Var': 'ω',
                        'Min': st.session_state["exp"].contents["output"]["omega_min"],
                        'Max': st.session_state["exp"].contents["output"]["omega_max"],
                    },
                ]
                st.table(st.session_state["table_data_min_max"])

                st.session_state["table_data_exp_std"] = []
                for name in st.session_state["config"]["data"]["names"]:
                    exp = st.session_state["exp"].contents["output"]["S"][name]["exp"]
                    std = st.session_state["exp"].contents["output"]["S"][name]["std"]
                    st.session_state["table_data_exp_std"].append({'Name': name, 'Expected Value': exp, 'Standard deviation': std})
                st.table(st.session_state["table_data_exp_std"])

            if "fig_data" in st.session_state:
                fig_data = st.session_state["fig_data"]
                pylit.ui.components.display_figure(fig_data)

        with tab4:
            # Create tabs for each data name
            tabs = st.tabs([ f"**{data_name}**" for data_name in st.session_state["config"]["data"]["names"]])

            for i in range(N): 
                with tabs[i]: 
                    # Create columns
                    col11, col12 = st.columns([1,1])

                    with col11:
                        # Method
                    
                        st.markdown(f"""<br/><b>Method</b><hr style='margin:0;padding:0'/>""", unsafe_allow_html=True)
                        pylit.ui.components.name_params(
                            key=f"method",
                            idx=i,
                            options=pylit.ui.settings.METHODS,
                            ref=pylit.methods,
                            param_map=pylit.ui.settings.METHODS_PARAM_MAP,
                        )

                    with col12:
                        # --- Optimizer --- # TODO CHANGE TO name_params with param_map ignore attributes 
                        st.markdown("<br/><b>Optimizer</b><hr style='margin:0;padding:0'/>", unsafe_allow_html=True)
                        st.selectbox(
                            label=f"Name",
                            options=pylit.ui.settings.OPTIMIZER(name=True),
                            key=f"optim_name_{i}",
                            index=pylit.ui.settings.OPTIMIZER.index_of(st.session_state["config"]["optim"]["name"][i]),
                            )
                        st.session_state["config"]["optim"]["name"][i] = pylit.ui.settings.OPTIMIZER.find(st.session_state[f"optim_name_{i}"]).ref

                        # Methodless optimizer attributes

                        for opt_name, opt_param in pylit.ui.settings.OPTIM_PARAM_MAP.items():
                            opt_key = f"ml{i}_{opt_name}"
                            st.session_state["config"]["optim"][opt_name][i] = pylit.ui.components.input(
                                param_name = opt_param.label,
                                param_type = opt_param.type,
                                param_default = opt_param.default,
                                id = opt_key,
                            )
                        # --- --- --- #
                    

                    # Model
                    st.markdown(f"""<br/><b>Model</b><hr style='margin:0;padding:0'/>""", unsafe_allow_html=True)
                    
                    pylit.ui.components.name_params(
                        key=f"scaling",
                        idx=i,
                        options=pylit.ui.settings.SCALINGS,
                        ref=pylit.models.lrm_scaling,
                        param_map=pylit.ui.settings.SCALINGS_PARAM_MAP,
                        label="Scaling",
                    )
                    pylit.ui.components.name_params(
                        key=f"model",
                        idx=i,
                        options=pylit.ui.settings.MODELS,
                        ref=pylit.models,
                        param_map=pylit.ui.param_map.ParamMap([
                            pylit.ui.param_map.Param(
                                name="omegas",
                                type=ARRAY,
                                default=[
                                    np.round(st.session_state["exp"].contents["output"]["omega_min"], 2),
                                    np.round(st.session_state["exp"].contents["output"]["omega_max"], 2),
                                    int(len(st.session_state["exp"].contents["output"]["S"]["omega"]["contents"])/5),
                                ],
                                min_value=-2*abs(st.session_state["exp"].contents["output"]["omega_min"]),
                                max_value=2*abs(st.session_state["exp"].contents["output"]["omega_max"]),
                            ),
                            pylit.ui.param_map.Param(
                                name="sigmas",
                                type=ARRAY,
                                default=[
                                    np.round(st.session_state["exp"].contents["output"]["S"][name]["std"], 2),
                                    np.round(10*st.session_state["exp"].contents["output"]["S"][name]["std"], 2),
                                    int(1/st.session_state["exp"].contents["output"]["S"][name]["std"]),
                                ],
                                min_value=1e-4,
                                max_value=20*st.session_state["exp"].contents["output"]["S"][name]["std"],
                            ),
                            pylit.ui.param_map.Param(
                                name="beta",
                                default=1.0,
                                ignore=True,
                            ),
                            pylit.ui.param_map.Param(
                                name="order",
                                default="0,1",
                                ignore=True,
                            ),
                        ]) if st.session_state["exp"] is not None else None
                    )

            # TODO: Change experiment s.t. the following projections are not necessary anymore
            st.session_state["config"]["method"]["name"] = [val["name"] for val in st.session_state["method"].values()]
            st.session_state["config"]["method"]["params"] = [val["params"] for val in st.session_state["method"].values()]        

            scaling = list(st.session_state["scaling"].values())
            model = list(st.session_state["model"].values())

            st.session_state["models"] = [{
                "name": name,
                "scaling": scaling[i],
                "model": model[i],
            } for i, name in enumerate(st.session_state["config"]["data"]["names"])]

            st.session_state["get_models"] = lambda exp, models=st.session_state["models"]: models # TODO: exp features?

        with tab5:
        # if st.session_state["nav"] == "JSON":
            if st.session_state["view_config_json"]:
                with st.expander("Config JSON", expanded=False):
                    st.write(st.session_state["config"])
            if st.session_state["view_models_json"]:
                with st.expander("Models JSON", expanded=False):
                    st.write(st.session_state["models"])
            if not st.session_state["view_config_json"] and not st.session_state["view_models_json"]:
                st.warning('Activate a JSON option in the settings.', icon="⚠️")

        with tab6:
            if st.button("Fit model"):
                with st.spinner('Fitting model ...'):
                    # TODO !
                    if st.session_state["preprocessed"]:
                        # Set config again
                        st.session_state["exp"].config = st.session_state["config"]
                        # TODO
                        st.session_state["fig_coeffs"], st.session_state["fig_results"]\
                            = st.session_state["exp"].fit_model(
                                st.session_state["get_models"],
                                store=st.session_state["store"],
                                docx=st.session_state["docx"],
                                giveback=True,
                                interactive=st.session_state["interactive"]
                            )
                        st.session_state["report"] = st.session_state["exp"].contents["report"]
                    else:
                        st.warning('Data not prepared yet. Please go to the Preprocessing tab and preprocess the data first.', icon="⚠️")
            st.markdown("<hr style='margin:0;padding:0'/>", unsafe_allow_html=True)
            if st.session_state["view_report"]:
                with st.expander("Report", expanded=True):
                    if "report" in st.session_state:
                        st.write(st.session_state["report"])                
            if st.session_state["view_coeffs_plot"]:
                with st.expander("Coefficients Plot", expanded=True):
                    if "fig_coeffs" in st.session_state:
                        fig_coeffs = st.session_state["fig_coeffs"]
                        pylit.ui.components.display_figure(fig_coeffs)
            with st.expander("Results Plot", expanded=True):
                if "fig_results" in st.session_state:
                    fig_results = st.session_state["fig_results"]
                    pylit.ui.components.display_figure(fig_results)

if __name__ == "__main__":
    main()