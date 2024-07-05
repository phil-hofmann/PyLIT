import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from pylit import ui
from pylit.global_settings import FLOAT_DTYPE, ARRAY, INT_DTYPE
from pylit.ui.options import Options
from pylit.ui.param_map import ParamMap
from pylit.core import DataLoader


def file_menu():
    with st.sidebar:
        st.button("📂 Open Experiment")


def settings_manager():
    if not ui.components.state_manager():
        if not os.path.exists(ui.settings.PATH_SETTINGS):
            ui.utils.save_settings(
                ui.settings.PATH_SETTINGS,
                ui.utils.load_settings(st.session_state),
            )
        else:
            st.session_state.update(
                ui.utils.load_session_state(ui.settings.PATH_SETTINGS)
            )


def state_manager() -> bool:
    exec_state = True
    if "workspace" not in st.session_state:
        st.session_state["workspace"] = ui.settings.PATH_PROJECT
        exec_state = False
    if "view_coeffs_plot" not in st.session_state:
        st.session_state["view_coeffs_plot"] = True
        exec_state = False
    if "wide_mode" not in st.session_state:
        st.session_state["wide_mode"] = True
        exec_state = False
    return exec_state


def reset_settings():
    st.session_state["workspace"] = ui.settings.PATH_PROJECT
    st.session_state["view_coeffs_plot"] = True
    st.session_state["wide_mode"] = True


def input(
    param_name: str,
    param_type,
    id: str,
    param_default=None,
    param_min=None,
    param_max=None,
    param_connect=None,
):
    # TODO pass param instead of the individual values
    if (
        param_type == int
        or param_type == INT_DTYPE
        or param_type == float
        or param_type == FLOAT_DTYPE
    ):

        min_value = (
            param_min
            if param_min is not None
            else 0.0 if param_type == float or param_type == FLOAT_DTYPE else 0
        )
        max_value = (
            param_max
            if param_max is not None
            else (
                5.0 * float(param_default)
                if (param_type == float or param_type == FLOAT_DTYPE)
                and param_default is not None
                else (
                    5 * int(param_default)
                    if (param_type == int or param_type == INT_DTYPE)
                    and param_default is not None
                    else None
                )
            )
        )
        default_value = (
            None
            if param_default is None
            else (
                float(param_default)
                if param_type == float or param_type == FLOAT_DTYPE
                else (
                    int(param_default)
                    if param_type == int or param_type == INT_DTYPE
                    else param_default
                )
            )
        )

        if st.session_state["sliders"]:
            return st.slider(
                label=param_name,
                min_value=min_value,
                max_value=max_value,
                value=default_value,
                step=(
                    ui.settings.NUM_STEP
                    if param_type == float or param_type == FLOAT_DTYPE
                    else 1
                ),
                key=id,
            )
        else:
            return st.number_input(
                label=param_name,
                value=default_value,
                step=(
                    ui.settings.SCI_NUM_STEP
                    if param_type == float or param_type == FLOAT_DTYPE
                    else 1
                ),
                format=(
                    "%f" if param_type == float or param_type == FLOAT_DTYPE else "%d"
                ),
                key=id,
            )

    elif param_type == str:
        return st.text_input(label=param_name, value=param_default, key=id)

    elif param_type == bool:
        return st.toggle(label=param_name, value=param_default, key=id)

    elif param_type == ARRAY:
        with st.expander(label=param_name, expanded=False):

            if st.session_state["sliders"]:

                col1, col2 = st.columns(2)
                with col1:
                    seq_type = st.selectbox(
                        label="Sequence Type", options=["Linear"], key=f"{id}_seq_type"
                    )

                with col2:
                    num_points = st.number_input(
                        label="Number of Points",
                        value=(
                            param_default[2]
                            if param_default is not None and len(param_default) > 2
                            else 10
                        ),
                        min_value=1,
                        step=1,
                        key=f"{id}_num_points",
                    )

                lower, upper = st.slider(
                    label="Interval",
                    step=ui.settings.NUM_STEP,
                    min_value=param_min if param_min is not None else 0.0,
                    max_value=(
                        param_max
                        if param_max is not None
                        else (
                            2.0 * param_default[1] if param_default is not None else 5.0
                        )
                    ),
                    value=(
                        (param_default[0], param_default[1])
                        if param_default is not None
                        else (0.0, 1.0)
                    ),
                    key=f"{id}_interval",
                )

            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    seq_type = st.selectbox(
                        label="Sequence Type", options=["Linear"], key=f"{id}_seq_type"
                    )

                with col2:
                    lower = st.number_input(
                        label="Lower Bound",
                        value=param_default[0] if param_default is not None else 0.0,
                        step=ui.settings.SCI_NUM_STEP,
                        format="%f",
                        key=f"{id}_lower",
                    )

                with col3:
                    upper = st.number_input(
                        label="Upper Bound",
                        value=param_default[1] if param_default is not None else 1.0,
                        step=ui.settings.SCI_NUM_STEP,
                        format="%f",
                        key=f"{id}_upper",
                    )

                with col4:
                    num_points = st.number_input(
                        label="Number of Points",
                        value=param_default[2] if param_default is not None else 10,
                        min_value=1,
                        step=1,
                        key=f"{id}_num_points",
                    )

            if seq_type == "Linear":
                return np.linspace(lower, upper, num_points)

            else:
                raise ValueError("Invalid sequence type specified.")
    else:
        raise ValueError(f"Unsupported parameter type: {param_type}")


def name_params( # TODO put to atoms
    my_id: str,
    options: Options,
    ref,
    param_map: ParamMap = None,
    label: str = "Name",
):
    if my_id not in st.session_state:
        st.session_state[my_id] = {"name": "", "params": {}}

    # Select name
    st.selectbox(
        label=label,
        options=options(name=True),
        placeholder=f"Select an option",
        index=0,
        key=f"{my_id}_name",
    )
    selected_value = st.session_state[f"{my_id}_name"]
    # Translate selected value to reference
    st.session_state[my_id]["name"] = options.find(selected_value).ref

    if st.session_state[my_id]["name"] != "":
        # Clear
        st.session_state[my_id]["params"] = {}
        # Extract parameters
        params = ui.utils.extract_params(
            func=getattr(ref, st.session_state[my_id]["name"])
        )
        for name, param in params.items():
            mapped_param = None
            if param_map is not None and name in param_map:
                mapped_param = param_map[name]
            display = True
            if mapped_param is not None:
                if mapped_param.ignore:
                    st.session_state[my_id]["params"][
                        name
                    ] = mapped_param.default  # NOTE dangerous // ignore
                    display = False

                if mapped_param.optional:
                    st.toggle(
                        label=mapped_param.optional_label,
                        key=f"{my_id}_{name}_optional",
                    )

            if display:
                if mapped_param is not None and (
                    not mapped_param.optional
                    or st.session_state[f"{my_id}_{name}_optional"]
                ):
                    st.session_state[my_id]["params"][name] = input(
                        param_name=mapped_param.name,
                        param_type=mapped_param.type,
                        param_default=mapped_param.default,
                        id=f"{my_id}_{name}",
                        param_min=mapped_param.min_value,
                        param_max=mapped_param.max_value,
                    )
                else:
                    st.session_state[my_id]["params"][name] = input(
                        param_name=name,
                        param_type=param.type,
                        param_default=param.default,
                        id=f"{my_id}_{name}",
                    )
    return st.session_state[my_id]["name"], st.session_state[my_id]["params"]


def display_figure(fig): # TODO put to atoms // Do I need that here???
    if isinstance(fig, list) and all(ui.utils.is_plotly_figure(fig) for fig in fig):
        num_figures = len(fig)
        num_rows = (num_figures + 1) // 2  # Add 1 to handle odd number of figures

        # Create subplots
        for i in range(num_rows):
            lb, ub = 2 * i, 2 * i + 1
            cols = st.columns(2)
            with cols[0]:
                st.plotly_chart(fig[lb], use_container_width=True)
            if ub < num_figures:
                with cols[1]:
                    st.plotly_chart(fig[ub], use_container_width=True)
    else:
        st.pyplot(fig)
