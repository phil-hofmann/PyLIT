import inspect
import json
import os
import streamlit as st
import plotly.graph_objects as go
from pylit.frontend.core import ParamMap, Param
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE
from pylit.frontend.constants import (
    PATH_SETTINGS,
    PATH_PROJECT,
)


def settings_manager():
    state_manager()
    if not os.path.exists(PATH_SETTINGS):
        save_settings(
            PATH_SETTINGS,
            load_settings(st.session_state),
        )
    else:
        st.session_state.update(load_session_state(PATH_SETTINGS))


def state_manager():
    if "workspace_as" not in st.session_state:
        st.session_state["workspace_as"] = PATH_PROJECT
    if "wide_mode_as" not in st.session_state:
        st.session_state["wide_mode_as"] = False
    if "auto_expand_as" not in st.session_state:
        st.session_state["auto_expand_as"] = True
    if "non_negative_as" not in st.session_state:
        st.session_state["non_negative_as"] = True
    if "detailed_balance_as" not in st.session_state:
        st.session_state["detailed_balance_as"] = True
    if "normalization_as" not in st.session_state:
        st.session_state["normalization_as"] = True
    if "time_scaling_as" not in st.session_state:
        st.session_state["time_scaling_as"] = True
    if "coefficients_as" not in st.session_state:
        st.session_state["coefficients_as"] = False
    if "default_model_as" not in st.session_state:
        st.session_state["default_model_as"] = True
    if "forward_default_model_as" not in st.session_state:
        st.session_state["forward_default_model_as"] = True
    if "forward_default_model_error_as" not in st.session_state:
        st.session_state["forward_default_model_error_as"] = False
    if "noise_samples_as" not in st.session_state:
        st.session_state["noise_samples_as"] = False
    if "model_as" not in st.session_state:
        st.session_state["model_as"] = True
    if "forward_model_as" not in st.session_state:
        st.session_state["forward_model_as"] = True
    if "forward_model_error_as" not in st.session_state:
        st.session_state["forward_model_error_as"] = False


def reset_settings():
    st.session_state["workspace_as"] = PATH_PROJECT
    st.session_state["wide_mode_as"] = False
    st.session_state["auto_expand_as"] = True
    st.session_state["non_negative_as"] = True
    st.session_state["detailed_balance_as"] = True
    st.session_state["normalization_as"] = True
    st.session_state["time_scaling_as"] = True
    st.session_state["default_model_as"] = True
    st.session_state["forward_default_model_as"] = True
    st.session_state["forward_default_model_error_as"] = True
    st.session_state["noise_samples_as"] = False
    st.session_state["coefficients_as"] = False
    st.session_state["model_as"] = True
    st.session_state["forward_model_as"] = True
    st.session_state["model_error_as"] = False
    st.session_state["forward_model_error_as"] = False


def is_data_file(file_path: str):
    """
    Check if the file is a data file based on its extension.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file is a data file, False otherwise.
    """
    # List of allowed data file extensions
    allowed_extensions = [".dat", ".csv", ".xlsx", ".xls", ".json", ".xml", ".tsv"]

    # Extract the file extension from the file path
    _, extension = os.path.splitext(file_path)

    # Check if the file extension is in the list of allowed extensions
    return extension.lower() in allowed_extensions


def load_settings(session_state):
    settings = {}
    keys = [  # TODO put setting keys into frontend/settings.py
        "workspace_as",
        "wide_mode_as",
        "auto_expand_as",
        "non_negative_as",
        "detailed_balance_as",
        "normalization_as",
        "time_scaling_as",
        "coefficients_as",
        "default_model_as",
        "forward_default_model_as",
        "forward_default_model_error_as",
        "noise_samples_as",
        "model_as",
        "forward_model_as",
        "forward_model_error_as",
    ]
    for key in keys:
        if key in session_state:
            settings[key] = session_state[key]
    return settings


def save_settings(filename, settings) -> bool:
    if not os.path.exists(settings.get("workspace_as")):
        return False
    with open(filename, "w") as file:
        json.dump(settings, file)
    return True


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
