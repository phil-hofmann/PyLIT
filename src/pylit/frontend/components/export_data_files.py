import os
import streamlit as st
from pylit.frontend.components.atoms import (
    FileSelector,
    ColumnDataExporter,
    PlotlyDataColumns,
    ShowDataTable,
)


def ExportDataFiles(my_id: str, default_directory: str, export_path: str):

    my_id_file_selector = f"{my_id}_file_selector"
    my_id_column_data_exporter = f"{my_id}_column_data_exporter"
    my_id_plotly_data_columns = f"{my_id}_plotly_data_columns"
    my_id_show_data_table = f"{my_id}_show_data_table"
    my_id_unselect_file_button = f"{my_id}_unselect_file_button"
    my_id_data_representation = f"{my_id}_data_representation"
    my_id_plotly_data_columns_button = f"{my_id}_plotly_data_columns_button"
    my_id_show_data_table_button = f"{my_id}_show_data_table_button"

    if my_id_column_data_exporter not in st.session_state:
        st.session_state[my_id_column_data_exporter] = os.path.isfile(export_path)

    if (
        my_id_file_selector in st.session_state
        and my_id_unselect_file_button in st.session_state
        and st.session_state[my_id_unselect_file_button]
    ):
        st.session_state[my_id_file_selector] = default_directory

    file_path = FileSelector(
        my_id=my_id_file_selector,
        default_directory=(
            export_path if os.path.isfile(export_path) else default_directory
        ),
    )

    if os.path.isfile(file_path):
        cols = st.columns([90, 10])

        data_exported = st.session_state[my_id_column_data_exporter]

        with cols[0]:
            if data_exported:
                st.success(
                    f"✅ &nbsp;  Successfully exported selected columns to '{export_path}'."
                )
            else:
                st.success(f"✅ &nbsp; Successfully fetched file.")
                st.warning(
                    f"⚠️ &nbsp; Make sure that you select two colums for the data to be exported."
                )

        with cols[1]:
            st.button("❌", key=my_id_unselect_file_button)
            st.session_state[my_id_data_representation] = ""

        if not data_exported:

            ColumnDataExporter(
                my_id=my_id_column_data_exporter,
                file_path=file_path,
                export_path=export_path,
            )

        st.markdown("<hr style='margin:0;padding:0;'/>", unsafe_allow_html=True)

        cols = st.columns([20, 20, 60])

        if my_id_data_representation not in st.session_state:
            st.session_state[my_id_data_representation] = ""

        with cols[0]:
            st.button(
                label="Plot Columns",
                key=my_id_plotly_data_columns_button,
            )

        if st.session_state[my_id_plotly_data_columns_button]:
            st.session_state[my_id_data_representation] = "plot"

        with cols[1]:
            st.button(
                label="Show Columns",
                key=my_id_show_data_table_button,
            )

        if st.session_state[my_id_show_data_table_button]:
            st.session_state[my_id_data_representation] = "show"

        if st.session_state[my_id_data_representation] == "plot":
            PlotlyDataColumns(
                my_id=my_id_plotly_data_columns,
                file_path=file_path,
            )
        elif st.session_state[my_id_data_representation] == "show":
            ShowDataTable(
                my_id=my_id_show_data_table,
                file_path=file_path,
            )
    return st.session_state[my_id_column_data_exporter]
