import numpy as np
import streamlit as st

from pylit.frontend.core import Param
from pylit.frontend.constants import SCI_NUM_STEP
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE


def Input(
    my_id: str,
    param: Param,
):
    if param.my_type in [INT_DTYPE, int, FLOAT_DTYPE, float] or type(param.my_type) in [
        INT_DTYPE,
        int,
        FLOAT_DTYPE,
        float,
    ]:
        attributes = param.attributes

        if "value" in attributes and type(attributes["value"]) in [ARRAY, list]:
            param.value = param.upper_value
            attributes["value"] = param.upper_value

        # Remove min_value, max_value, and step from attributes
        if "lower_value" in attributes:
            del attributes["lower_value"]
        if "upper_value" in attributes:
            del attributes["upper_value"]
        if "num_points" in attributes:
            del attributes["num_points"]

        return st.number_input(
            key=my_id,
            **attributes,
        )
    elif param.my_type == str:
        value = param.value
        if type(param.value) is not str:
            value = ""
        return st.text_input(
            key=my_id,
            label=param.label,
            value=value,
        )
    elif param.my_type == bool:
        value = param.value
        if type(param.value) is not bool:
            value = False
        if param.description is None:
            return st.toggle(
                key=my_id,
                label=param.label,
                value=value,
            )
        else:
            col = st.columns([1, 2], vertical_alignment="center")
            with col[0]:
                toggle_component = st.toggle(
                    key=my_id,
                    label=param.label,
                    value=value,
                )
            with col[1]:
                st.markdown(param.description)
            return toggle_component

    elif param.my_type in [list, ARRAY]:
        my_id_seq_type = f"{my_id}_seq_type"
        my_id_lower = f"{my_id}_lower"
        my_id_upper = f"{my_id}_upper"
        my_id_num_points = f"{my_id}_num_points"
        lower_value, upper_value, num_value = param.default  # TODO Handle that
        with st.expander(
            label=param.label, expanded=st.session_state["auto_expand_as"]
        ):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                seq_type = st.selectbox(
                    label="Type", options=["Linear"], key=my_id_seq_type
                )  # TODO For several types need to change this!!!
            with col2:
                lower = st.number_input(
                    key=my_id_lower,
                    label="Lower",
                    value=lower_value,
                    step=SCI_NUM_STEP,
                    format="%f",
                )
            with col3:
                upper = st.number_input(
                    key=my_id_upper,
                    label="Upper",
                    value=upper_value,
                    step=SCI_NUM_STEP,
                    format="%f",
                )
            with col4:
                num_points = st.number_input(
                    key=my_id_num_points,
                    label="Size",
                    value=num_value,
                    min_value=1,
                    step=1,
                )
            # TODO Provide more sequence types
            if seq_type == "Linear":
                return np.linspace(lower, upper, num_points)
            else:
                raise ValueError("Invalid sequence type specified.")
    else:
        raise ValueError(f"Unsupported parameter type: {param.my_type}")
