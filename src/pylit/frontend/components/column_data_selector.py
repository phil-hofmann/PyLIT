import pandas as pd
import streamlit as st


def ColumnDataSelector(
    my_id: str,
    first_var_label: str,
    second_var_label: str,
    data: pd.DataFrame,
):
    if my_id not in st.session_state:
        st.session_state[my_id] = False

    my_id_radio_first_var = f"{my_id}_checkbox_first_var"
    my_id_radio_second_var = f"{my_id}_checkbox_second_var"
    my_id_selected_columns = f"{my_id}_selected_columns"

    if len(data.shape) < 2:
        st.error("❌ &nbsp; Please select a file with at least two columns.")
        return False

    st.info(
        "Make sure that you select two colums for the data to be exported.", icon="ℹ️"
    )

    num_columns = data.shape[1]

    # Create a label for the first variable
    label_first_var = "**Select " + first_var_label + "**"
    # Create the options for the first variable
    options_first_var = [f"Column {i}" for i in range(num_columns)]
    # Create radio buttons for the first variable
    st.radio(
        label=label_first_var,
        options=options_first_var,
        index=0,
        key=my_id_radio_first_var,
        horizontal=True,
    )

    # Create a label for the second variable
    label_second_var = "**Select " + second_var_label + "**"
    # Create the options for the second variable
    options_second_var = [f"Column {i}" for i in range(num_columns)]
    # Create checkboxes for each column for the second variable
    st.radio(
        label=label_second_var,
        options=options_second_var,
        index=1,
        key=my_id_radio_second_var,
        horizontal=True,
    )

    st.session_state[my_id_selected_columns] = [
        options_first_var.index(st.session_state[my_id_radio_first_var]),
        options_second_var.index(st.session_state[my_id_radio_second_var]),
    ]

    return st.session_state[my_id_selected_columns]
