import inspect
import json
import os
import streamlit as st
import plotly.graph_objects as go
from pylit.frontend.core import ParamMap, Param
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE
from pylit.frontend.constants import(
    PATH_SETTINGS,
    PATH_PROJECT,
)

def settings_manager():
    if state_manager():
        if not os.path.exists(PATH_SETTINGS):
            save_settings(
                PATH_SETTINGS,
                load_settings(st.session_state),
            )
        else:
            st.session_state.update(
                load_session_state(PATH_SETTINGS)
            )


def state_manager() -> bool:
    exec_state = True
    if "workspace" not in st.session_state:
        st.session_state["workspace"] = PATH_PROJECT
        exec_state = False
    if "view_coeffs_plot" not in st.session_state:
        st.session_state["view_coeffs_plot"] = True
        exec_state = False
    if "wide_mode" not in st.session_state:
        st.session_state["wide_mode"] = True
        exec_state = False
    return exec_state


def reset_settings():
    st.session_state["workspace"] = PATH_PROJECT
    st.session_state["view_coeffs_plot"] = True
    st.session_state["wide_mode"] = True

def is_data_file(file_path: str):
    """
    Check if the file is a data file based on its extension.
    
    Args:
        file_path (str): The path to the file.
        
    Returns:
        bool: True if the file is a data file, False otherwise.
    """
    # List of allowed data file extensions
    allowed_extensions = ['.dat', '.csv', '.xlsx', '.xls', '.json', '.xml', '.tsv']
    
    # Extract the file extension from the file path
    _, extension = os.path.splitext(file_path)
    
    # Check if the file extension is in the list of allowed extensions
    return extension.lower() in allowed_extensions

def load_settings(session_state):
    settings = {}
    keys = [ # TODO put setting keys into frontend/settings.py
        "workspace",
        "view_coeffs_plot",
        "wide_mode",
    ]
    for key in keys:
        if key in session_state:
            settings[key] = session_state[key]
    return settings


def save_settings(filename, settings):
    with open(filename, "w") as file:
        json.dump(settings, file)


def load_session_state(filename):
    try:
        with open(filename, "r") as file:
            body = json.load(file)
            return body
    except FileNotFoundError:
        return {}


def is_plotly_figure(figure):
    return isinstance(figure, go.Figure)

def extract_params(func: callable) -> ParamMap:
    param_list = []
    params = inspect.signature(func).parameters
    for name, param in params.items():
        if param.annotation != inspect.Parameter.empty:
            param_type = param.annotation
            if (
                param_type == str
                or param_type == int
                or param_type == float
                or param_type == bool
                or param_type == list
                or param_type == dict
                or param_type == tuple
                or param_type == set
                or param_type == ARRAY
                or param_type == FLOAT_DTYPE
                or param_type == INT_DTYPE
            ):
                param_default = (
                    param.default if param.default != inspect.Parameter.empty else None
                )
                param_list.append(
                    Param(name=name, my_type=param_type, default=param_default)
                )
    return ParamMap(param_list)