import inspect
import json
import os
import plotly.graph_objects as go
from pylit.ui.param_map import ParamMap, Param
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE

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
    keys = [ # TODO put setting keys into ui/settings.py
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
        print(f"settings = {settings}")
        json.dump(settings, file)


def load_session_state(filename):
    try:
        with open(filename, "r") as file:
            body = json.load(file)
            print(f"body = {body}")
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
                    Param(name=name, type=param_type, default=param_default)
                )
    return ParamMap(param_list)


if __name__ == "__main__":
    pass
