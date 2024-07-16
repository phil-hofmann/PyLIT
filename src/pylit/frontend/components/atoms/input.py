import numpy as np
import streamlit as st

from pylit.frontend.core import Param
from pylit.frontend.constants import SCI_NUM_STEP
from pylit.global_settings import ARRAY, FLOAT_DTYPE, INT_DTYPE

def Input(
    my_id: str,
    param: Param,
):
    if param.my_type in [INT_DTYPE, int, FLOAT_DTYPE, float]:
        return st.number_input(
            key=my_id,
            **param.attributes,
        )
    elif param.my_type == str:
        return st.text_input(
            key=my_id,
            label=param.label,
            value=param.value, 
        )
    elif param.my_type == bool:
        return st.toggle(
            key=my_id,
            label=param.label,
            value=param.value,
        )
    elif param.my_type in [list, ARRAY]:
        my_id_seq_type = f"{my_id}_seq_type"
        my_id_lower = f"{my_id}_lower"
        my_id_upper = f"{my_id}_upper"
        my_id_num_points = f"{my_id}_num_points"
        lower_value, upper_value, num_value = param.default # TODO Handle that
        with st.expander(label=param.label, expanded=True):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1 ])
            with col1:
                seq_type = st.selectbox(
                    label="Type", options=["Linear"], key=my_id_seq_type
                )
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