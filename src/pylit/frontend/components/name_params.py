import streamlit as st

from pylit.global_settings import ARRAY
from pylit.frontend.utils import extract_params
from pylit.frontend.core import Options, ParamMap
from pylit.frontend.components.atoms import Input
from pylit.global_settings import FLOAT_DTYPE


def NameParams(
    my_id: str,
    options: Options,
    ref,
    param_map: ParamMap = None,
    label: str = "Name",
    name: str="",
):

    my_id_name = f"{my_id}_name"

    if my_id not in st.session_state:
        st.session_state[my_id] = {"name": name, "params": {}}

    # Select a name
    st.selectbox(
        label=label,
        options=options(name=True),
        placeholder=f"Select an option",
        index=options.index_of(name),
        key=my_id_name,
    )

    # Translate selected value to reference class
    st.session_state[my_id]["name"] = options.find(st.session_state[my_id_name]).ref

    if st.session_state[my_id]["name"] != "":
        # Extract parameters from reference
        params = extract_params(func=getattr(ref, st.session_state[my_id]["name"]))

        for param_name, param in params.items():
            my_id_param = f"{my_id}_{param_name}"
            if param_map is not None and param_name in param_map:
                mapped_param = param_map[param_name]
                my_id_param_variation = f"{my_id_param}_variation"
                if mapped_param.ignore:
                    st.session_state[my_id]["params"][param_name] = mapped_param.default
                elif mapped_param.variation:
                    toggle_value = mapped_param.value is not None and type(mapped_param.value) not in [FLOAT_DTYPE, float]
                    st.toggle(
                        key=my_id_param_variation,
                        label=f"Variation of {mapped_param.name}",
                        value=toggle_value,
                    )
                    mapped_param.my_type = (
                        ARRAY
                        if st.session_state[my_id_param_variation]
                        else mapped_param.my_type
                    )
                if not mapped_param.ignore:
                    st.session_state[my_id]["params"][param_name] = Input(
                        my_id=my_id_param,
                        param=mapped_param,
                    )
            else:
                st.session_state[my_id]["params"][param_name] = Input(
                    my_id=my_id_param,
                    param=param,
                )

    return st.session_state[my_id]["name"], st.session_state[my_id]["params"]
