import streamlit as st

from pylit.frontend.utils import extract_params
from pylit.frontend.core import Options, ParamMap
from pylit.frontend.components.hr import HR
from pylit.frontend.components.input import Input


def NameParams(
    _selected_ref,
    my_id: str,
    options: Options,
    ref,
    param_map: ParamMap = None,
    label: str = "Name",
    name: str = "",
):

    my_id_name = f"{my_id}_name"

    selected_name = st.selectbox(
        label=label,
        options=options(name=True),
        placeholder="Select an option",
        index=options.index_of(name),
        key=my_id_name,
    )

    _selected_ref = options.find(selected_name).ref

    # Display parameters section if a valid name is selected
    HR()
    st.markdown("**Parameters**")

    params_dict = {}

    if _selected_ref != "":
        # Extract parameters from the reference class dynamically
        params = extract_params(func=getattr(ref, _selected_ref))

        # TODO Check the following code
        # Iterate over each parameter and create input fields
        for param_name, param in params.items():
            my_id_param = f"{my_id}_{param_name}"

            if param_map is not None and param_name in param_map:
                mapped_param = param_map[param_name]
                if mapped_param.ignore:
                    params_dict[param_name] = mapped_param.default
                else:
                    # Render input component with session state, directly passing mapped params
                    params_dict[param_name] = Input(
                        my_id=my_id_param,
                        param=mapped_param,
                    )
            else:
                # Default case for handling parameters not in param_map
                params_dict[param_name] = Input(
                    my_id=my_id_param,
                    param=param,
                )

    return params_dict


# def NameParams(
#     my_id: str,
#     options: Options,
#     ref,
#     param_map: ParamMap = None,
#     label: str = "Name",
#     name: str = "",
# ):

#     my_id_name = f"{my_id}_name"

#     selected_name = st.selectbox(
#         label=label,
#         options=options(name=True),
#         placeholder="Select an option",
#         index=options.index_of(name),
#         key=my_id_name,
#     )

#     selected_ref = options.find(selected_name).ref

#     # Display parameters section if a valid name is selected
#     HR()
#     st.markdown("**Parameters**")

#     params_dict = {}

#     if selected_ref != "":
#         # Extract parameters from the reference class dynamically
#         params = extract_params(func=getattr(ref, selected_ref))

#         # TODO Check the following code
#         # Iterate over each parameter and create input fields
#         for param_name, param in params.items():
#             my_id_param = f"{my_id}_{param_name}"

#             if param_map is not None and param_name in param_map:
#                 mapped_param = param_map[param_name]
#                 if mapped_param.ignore:
#                     params_dict[param_name] = mapped_param.default
#                 else:
#                     # Render input component with session state, directly passing mapped params
#                     params_dict[param_name] = Input(
#                         my_id=my_id_param,
#                         param=mapped_param,
#                     )
#             else:
#                 # Default case for handling parameters not in param_map
#                 params_dict[param_name] = Input(
#                     my_id=my_id_param,
#                     param=param,
#                 )

#     return selected_ref, params_dict


# def NameParams(
#     my_id: str,
#     options: Options,
#     ref,
#     param_map: ParamMap = None,
#     label: str = "Name",
#     name: str="",
# ):

#     my_id_name = f"{my_id}_name"

#     if my_id not in st.session_state:
#         st.session_state[my_id] = {"name": name, "params": {}}

#     # Select a name
#     st.selectbox(
#         label=label,
#         options=options(name=True),
#         placeholder=f"Select an option",
#         index=options.index_of(name),
#         key=my_id_name,
#     )

#     # Translate selected value to reference class
#     st.session_state[my_id]["name"] = options.find(st.session_state[my_id_name]).ref
#     st.markdown("<hr style='margin: 0px;'>", unsafe_allow_html=True)
#     st.markdown("**Parameters**")
#     if st.session_state[my_id]["name"] != "":
#         # Extract parameters from reference
#         params = extract_params(func=getattr(ref, st.session_state[my_id]["name"]))

#         for param_name, param in params.items():
#             my_id_param = f"{my_id}_{param_name}"
#             if param_map is not None and param_name in param_map:
#                 mapped_param = param_map[param_name]
#                 my_id_param_variation = f"{my_id_param}_variation"
#                 if mapped_param.ignore:
#                     st.session_state[my_id]["params"][param_name] = mapped_param.default
#                 elif mapped_param.variation:
#                     toggle_value = mapped_param.value is not None and type(mapped_param.value) not in [FLOAT_DTYPE, float]
#                     # st.write("value = ", mapped_param.value) # TODO: Remove
#                     st.toggle(
#                         key=my_id_param_variation,
#                         label=f"Variation of {mapped_param.label}",
#                         value=toggle_value,
#                     )
#                     mapped_param.my_type = (
#                         ARRAY
#                         if st.session_state[my_id_param_variation]
#                         else mapped_param.my_type
#                     )
#                 if not mapped_param.ignore:
#                     st.session_state[my_id]["params"][param_name] = Input(
#                         my_id=my_id_param,
#                         param=mapped_param,
#                     )
#             else:
#                 st.session_state[my_id]["params"][param_name] = Input(
#                     my_id=my_id_param,
#                     param=param,
#                 )

#     return st.session_state[my_id]["name"], st.session_state[my_id]["params"]
