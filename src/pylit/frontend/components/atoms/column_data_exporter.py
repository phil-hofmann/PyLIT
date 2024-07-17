import streamlit as st
from pylit.backend.core import DataLoader


def ColumnDataExporter(my_id: str, file_path: str, export_path: str):
    if my_id not in st.session_state:
        st.session_state[my_id] = False

    my_id_button = f"{my_id}_button"
    my_id_checkbox = f"{my_id}_checkbox"

    dl = DataLoader(file_path=file_path)
    dl.fetch()
    num_columns = dl.data.shape[1]
    cols = st.columns(num_columns + 1)

    if "selected_columns" not in st.session_state:
        st.session_state["selected_columns"] = []

    # Button to export selected columns
    with cols[0]:
        st.button(
            "Export",
            key=my_id_button,
        )
    if st.session_state[my_id_button]:
        if len(st.session_state["selected_columns"]) == 2:
            # Use the dl.store method to store the selected columns
            dl.store(export_path, *st.session_state["selected_columns"])
            st.session_state[my_id] = True
        else:
            st.error("❌ &nbsp; Please select exactly two columns to export.")

    # Create checkboxes for each column
    for i in range(num_columns):
        with cols[i + 1]:
            my_id_checkbox_i = f"{my_id_checkbox}_{i}"
            st.checkbox(
                f"Column {i}",
                key=my_id_checkbox_i,
                value=False,
            )
            if (
                st.session_state[my_id_checkbox_i]
                and i not in st.session_state["selected_columns"]
            ):
                st.session_state["selected_columns"].append(i)
            elif (
                not st.session_state[my_id_checkbox_i]
                and i in st.session_state["selected_columns"]
            ):
                st.session_state["selected_columns"].remove(i)
    return st.session_state[my_id]
