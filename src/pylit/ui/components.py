import os
import pylit
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylit
import plotly.graph_objects as go

from pylit.global_settings import FLOAT_DTYPE, ARRAY, INT_DTYPE
from pylit.ui.settings import PATH_DATA
from pylit.ui.options import Options
from pylit.ui.param_map import ParamMap

def file_menu():
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            st.button("📂 Open")
        with col2:
            st.button("🗋 New")

def settings_manager():
    if not pylit.ui.components.state_manager():
        if not os.path.exists(pylit.ui.settings.PATH_SETTINGS):
            pylit.ui.utils.save_settings(
                pylit.ui.settings.PATH_SETTINGS,
                pylit.ui.utils.load_settings(st.session_state)
            )
        else:
            st.session_state.update(
                pylit.ui.utils.load_session_state(
                    pylit.ui.settings.PATH_SETTINGS
                )
            )

def state_manager() -> bool:
    exec_state = True
    if "workspace" not in st.session_state:
        st.session_state["workspace"] = pylit.ui.settings.PATH_EXPERIMENTS
        exec_state = False
    if "force" not in st.session_state:
        st.session_state["force"] = False
        exec_state = False
    if "interactive" not in st.session_state:
        st.session_state["interactive"] = True
        exec_state = False
    if "sliders" not in st.session_state:
        st.session_state["sliders"] = True
        exec_state = False
    if "store" not in st.session_state:
        st.session_state["store"] = True
        exec_state = False
    if "docx" not in st.session_state:
        st.session_state["docx"] = True
        exec_state = False
    if "view_config_json" not in st.session_state:
        st.session_state["view_config_json"] = True
        exec_state = False
    if "view_models_json" not in st.session_state:
        st.session_state["view_models_json"] = False
        exec_state = False
    if "view_report" not in st.session_state:
        st.session_state["view_report"] = False
        exec_state = False
    if "view_coeffs_plot" not in st.session_state:
        st.session_state["view_coeffs_plot"] = True
        exec_state = False
    if "wide_mode" not in st.session_state:
        st.session_state["wide_mode"] = True
        exec_state = False
    return exec_state

def reset_settings():
    st.session_state["workspace"] = pylit.ui.settings.PATH_EXPERIMENTS
    st.session_state["force"] = False
    st.session_state["interactive"] = True
    st.session_state["sliders"] = True
    st.session_state["store"] = True
    st.session_state["docx"] = True
    st.session_state["view_config_json"] = True
    st.session_state["view_models_json"] = True
    st.session_state["view_report"] = True
    st.session_state["view_coeffs_plot"] = True

def input(param_name: str, param_type, id: str,  param_default=None, param_min=None, param_max=None, param_connect=None):
    # TODO pass param instead of the individual values
    if param_type == int or param_type == INT_DTYPE \
        or param_type == float or param_type == FLOAT_DTYPE:

        min_value  = param_min if param_min is not None else 0.0 if param_type == float or param_type == FLOAT_DTYPE else 0
        max_value  = param_max if param_max is not None else 5.0 * float(param_default) if (param_type == float or param_type == FLOAT_DTYPE) and param_default is not None else 5 * int(param_default) if (param_type == int or param_type == INT_DTYPE) and param_default is not None else None
        default_value = None if param_default is None else float(param_default) if param_type == float or param_type == FLOAT_DTYPE else int(param_default) if param_type == int or param_type == INT_DTYPE else param_default

        if st.session_state["sliders"]:
            return st.slider(
                label=param_name,
                min_value=min_value,
                max_value=max_value,
                value=default_value,
                step=pylit.ui.settings.NUM_STEP if param_type == float or param_type == FLOAT_DTYPE else 1,
                key=id,
                )
        else:
            return st.number_input(
                label=param_name,
                value=default_value,
                step=pylit.ui.settings.SCI_NUM_STEP if param_type == float or param_type == FLOAT_DTYPE else 1,
                format="%f" if param_type == float or param_type == FLOAT_DTYPE else "%d",
                key=id,
                )
    
    elif param_type == str:
        return st.text_input(
            label=param_name,
            value=param_default,
            key=id
            )
    
    elif param_type == bool:
        return st.toggle(
            label=param_name,
            value=param_default,
            key=id
            )
    
    elif param_type == ARRAY:
        with st.expander(label=param_name, expanded=False):
            
            if st.session_state["sliders"]:

                col1, col2 = st.columns(2)
                with col1:
                    seq_type = st.selectbox(
                        label="Sequence Type", 
                        options=["Linear"], 
                        key=f"{id}_seq_type"
                        )
                
                with col2:
                    num_points = st.number_input(
                        label="Number of Points",
                        value=param_default[2] if param_default is not None and len(param_default) > 2 else 10,
                        min_value=1,
                        step=1,
                        key=f"{id}_num_points")
                    
                lower, upper = st.slider(
                    label="Interval",
                    step=pylit.ui.settings.NUM_STEP,
                    min_value=param_min if param_min is not None else 0.0,
                    max_value=param_max if param_max is not None else 2.0 * param_default[1] if param_default is not None else 5.0,
                    value=(param_default[0], param_default[1]) if param_default is not None else (0.0, 1.0),
                    key=f"{id}_interval",
                )

            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    seq_type = st.selectbox(
                        label="Sequence Type", 
                        options=["Linear"], 
                        key=f"{id}_seq_type"
                        )

                with col2:
                    lower = st.number_input(
                        label="Lower Bound",
                        value=param_default[0] if param_default is not None else 0.0, 
                        step=pylit.ui.settings.SCI_NUM_STEP,
                        format="%f",
                        key=f"{id}_lower"
                        )
                
                with col3:
                    upper = st.number_input(
                        label="Upper Bound",
                        value=param_default[1] if param_default is not None else 1.0,
                        step=pylit.ui.settings.SCI_NUM_STEP,
                        format="%f",
                        key=f"{id}_upper"
                        )
                
                with col4:
                    num_points = st.number_input(
                        label="Number of Points",
                        value=param_default[2] if param_default is not None else 10,
                        min_value=1,
                        step=1,
                        key=f"{id}_num_points")

            if seq_type == "Linear":
                return np.linspace(lower, upper, num_points)
            
            else:
                raise ValueError("Invalid sequence type specified.")
    else:
        raise ValueError(f"Unsupported parameter type: {param_type}")
    
def name_params(key: str, idx: int, options: Options, ref, param_map: ParamMap=None, label:str="Name"):
    if key not in st.session_state:
        st.session_state[key] = {}
    if idx not in st.session_state[key]:
        st.session_state[key][idx] = {
            "name":"", 
            "params":{}
            }
    # Select name
    st.selectbox(
        label=label,
        options=options(name=True),
        placeholder=f"Select an option",
        index=0,
        key=f"{key}_{idx}_name",
    )
    selected_value = st.session_state[f"{key}_{idx}_name"]
    # Translate selected value to reference
    st.session_state[key][idx]["name"] = options.find(selected_value).ref
    
    if st.session_state[key][idx]["name"] != "":
        # Clear
        st.session_state[key][idx]["params"] = {}
        # Extract parameters
        params = pylit.ui.utils.extract_params(
            func=getattr(ref, st.session_state[key][idx]["name"])
        )
        for name, param in params.items():
            mapped_param = None
            if param_map is not None and name in param_map:
                mapped_param = param_map[name]
            display = True
            if mapped_param is not None:
                if mapped_param.ignore:
                    st.session_state[key][idx]["params"][name] = mapped_param.default # NOTE dangerous // ignore
                    display = False

                if mapped_param.optional:
                    st.toggle(
                        label=mapped_param.optional_label,
                        key=f"{key}_{idx}_{name}_optional",
                        )

            if display:
                if mapped_param is not None and (not mapped_param.optional or st.session_state[f"{key}_{idx}_{name}_optional"]):
                    st.session_state[key][idx]["params"][name] = input(
                        param_name=mapped_param.name,
                        param_type=mapped_param.type,
                        param_default=mapped_param.default,
                        id=f"{key}_{idx}_{name}",
                        param_min=mapped_param.min_value,
                        param_max=mapped_param.max_value,
                    )
                else:
                    st.session_state[key][idx]["params"][name] = input(
                        param_name=name,
                        param_type=param.type,
                        param_default=param.default,
                        id=f"{key}_{idx}_{name}",
                    )
                    

def display_figure(fig):
    if isinstance(fig, list) and all(pylit.ui.utils.is_plotly_figure(fig) for fig in fig):
        num_figures = len(fig)
        num_rows = (num_figures + 1) // 2  # Add 1 to handle odd number of figures
        
        # Create subplots
        for i in range(num_rows):
            lb, ub = 2*i, 2*i+1
            cols = st.columns(2)
            with cols[0]:
                st.plotly_chart(fig[lb], use_container_width=True)
            if ub < num_figures:
                with cols[1]:
                    st.plotly_chart(fig[ub], use_container_width=True)
    else:
        st.pyplot(fig)

def file_selector(id: str, default_directory: str = PATH_DATA):

    if "file_selector_component" not in st.session_state:
        st.session_state["file_selector_component"] = {}

    if id not in st.session_state["file_selector_component"]:
        st.session_state["file_selector_component"][id] = default_directory

    def button_callback(value: str):
        st.session_state["file_selector_component"][id] = value

    def goback_callback():
        st.session_state["file_selector_component"][id] = os.path.dirname(st.session_state["file_selector_component"][id])

    st.write()

    col11, col12 = st.columns([6, 94])

    # Go back button
    with col11:
        st.markdown("<div style='margin-top:30px'/>", unsafe_allow_html=True)
        st.button(f"↩", key=f"{id}_gobackbtn",  on_click=goback_callback)
    # Directory selection widget
    with col12:
        st.text_input(label="Selected Path", key=id, value=st.session_state["file_selector_component"][id])
    
    st.markdown("<hr style='margin:0;padding:0;'/>", unsafe_allow_html=True)

    if os.path.isdir(st.session_state["file_selector_component"][id]):
        # Add custom CSS to change the width and remove border of specific buttons
        st.markdown("""
        <style>
        .button-after {
            display: none;
        }
        .element-container:has(.button-after) {
            display: none;
        }
        .element-container:has(.button-after) + div button {
            border: none;
            width: 100%;
            border-radius:0;
            text-align: left;
            margin:0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # List files and folders in the selected directory
        col11, col12 = st.columns([6, 94])

        with col12:
            with os.scandir(st.session_state["file_selector_component"][id]) as entries:
                for i, entry in enumerate(entries):
                    st.markdown('<span class="button-after"></span>', unsafe_allow_html=True)
                    if entry.is_dir():
                        st.button(f"📁 {entry.name}", key=f"{id}_btn_{i}", on_click=lambda x=os.path.join(st.session_state["file_selector_component"][id], entry.name): button_callback(x))
                    elif entry.is_file() and entry.name.endswith(".dat"):
                        st.button(f"📄 {entry.name}", key=f"{id}_btn_{i}", on_click=lambda x=os.path.join(st.session_state["file_selector_component"][id], entry.name): button_callback(x))
    
    if os.path.isfile(st.session_state["file_selector_component"][id]):
        dl = pylit.DataLoader(file_name=st.session_state["file_selector_component"][id])
        dl.fetch()
        amount = dl.data.shape[1] - pylit.ui.settings.IGNORE_FIRST_COLUMNS
        st.success(f'✅ Successfully selected a file.')
        st.warning(f'⚠️ Make sure that you name {amount} columns for the data to be loaded.')

        if st.button("Show Columns", key=f"{id}_plot_columns"):
            if "interactive" in st.session_state and st.session_state["interactive"]:
                plotly_columns(file_path=st.session_state["file_selector_component"][id])
            
            else:
                plot_columns(file_path=st.session_state["file_selector_component"][id])

    return st.session_state["file_selector_component"][id]

def plot_columns(file_path: str):
    # Load data using DataLoader
    loader = pylit.DataLoader(file_name=file_path)
    loader.fetch()
    
    # Plot each column
    num_columns = loader.data.shape[1]
    fig, axes = plt.subplots(num_columns, 1, figsize=(10, 5*num_columns))
    
    for i in range(num_columns):
        axes[i].plot(loader.data[:, i])
        axes[i].set_title(f"Column {i+1}")
        axes[i].set_xlabel("Index")
        axes[i].set_ylabel("Value")
    
    plt.tight_layout()
    st.pyplot(fig)

def plotly_columns(file_path: str):
    # Load data using DataLoader
    loader = pylit.DataLoader(file_name=file_path)
    loader.fetch()
    
    # Convert data to DataFrame
    df = pd.DataFrame(loader.data, columns=[f"Column {i+1}" for i in range(loader.data.shape[1])])
    
    # Create traces for each column
    data = []
    for i, column in enumerate(df.columns):
        trace = go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines',
            name=column,
            line=dict(color=pylit.ui.settings.PLOTLY_COLORS[i])
            )
        data.append(trace)
    
    # Create layout
    layout = go.Layout(
        title='Plot of Columns',
        xaxis=dict(title='Index'),
        yaxis=dict(title='Value'),
    )
    
    # Create figure
    fig = go.Figure(data=data, layout=layout)
    
    # Display the plotly chart
    st.plotly_chart(fig)

if __name__ == "__main__":

    # Example Usage

    ex_array = input(
        param_name=f"Example {ARRAY}",
        param_type=ARRAY,
        param_default=None,
        id="ex_array")
    st.write()

    ex_float = input(
        param_name=f"Example {FLOAT_DTYPE}",
        param_type=FLOAT_DTYPE,
        param_default=0.1,
        id="ex_float")
    st.write()
    
    ex_int = input(
        param_name=f"Example {INT_DTYPE}",
        param_type=INT_DTYPE,
        param_default=1,
        id="ex_int")
    st.write()

    ex_str = input(
        param_name="Example String",
        param_type=str,
        param_default="Hello World",
        id="ex_str")
    st.write()

    ex_bool = input(
        param_name="Example Boolean",
        param_type=bool,
        param_default=True,
        id="ex_bool")
    
    ex_file = file_selector(
        label="Select a file",
        id="ex_file",
        default_directory=PATH_DATA
        )
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    with st.expander(
        label="Show Inputs",
        expanded=True,
        ):

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.write("Example Array:")
            st.write(ex_array)
        with col2:
            st.write("Example Float:")
            st.write(ex_float)
        with col3:
            st.write("Example Int:")
            st.write(ex_int)
        with col4:
            st.write("Example String:")
            st.write(ex_str)
        with col5:
            st.write("Example Boolean:")
            st.write(ex_bool)
