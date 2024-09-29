import os
import streamlit as st

from streamlit import session_state as state
from pylit.backend.core import DataLoader
from pylit.frontend.components.hr import HR
from pylit.frontend.components.label import Label
from pylit.frontend.components.latex_row import LatexRow
from pylit.frontend.components.file_selector import FileSelector
from pylit.frontend.components.show_data_table import ShowDataTable
from pylit.frontend.components.plotly_data_columns import PlotlyDataColumns
from pylit.frontend.components.column_data_selector import ColumnDataSelector


def ExportDataFiles(
    my_id: str,
    first_var_label: str,
    second_var_label: str,
    default_directory: str,
    export_path: str,
):

    my_id_file_selector = f"{my_id}_file_selector"
    my_id_column_data_exporter = f"{my_id}_column_data_exporter"
    my_id_column_data_exporter_button = f"{my_id}_column_data_exporter_button"
    my_id_selected_columns = f"{my_id}_selected_columns"
    my_id_plotly_data_columns = f"{my_id}_plotly_data_columns"
    my_id_show_data_table = f"{my_id}_show_data_table"
    my_id_unselect_file_button = f"{my_id}_unselect_file_button"
    my_id_data_representation = f"{my_id}_data_representation"
    my_id_plotly_data_columns_button = f"{my_id}_plotly_data_columns_button"
    my_id_show_data_table_button = f"{my_id}_show_data_table_button"

    # Initialize session state
    if my_id_column_data_exporter not in state:
        # Check if the data has been exported
        state[my_id_column_data_exporter] = os.path.isfile(export_path)

    if my_id_data_representation not in state:
        state[my_id_data_representation] = ""

    # - Unselect File Button Clicked
    if (
        my_id_file_selector in state
        and my_id_unselect_file_button in state
        and state[my_id_unselect_file_button]
    ):
        state[my_id_column_data_exporter] = False  # Reset the data_exported state
        state[my_id_file_selector] = default_directory  # Reset the file selector
        state[my_id_data_representation] = ""  # Reset the data representation

    # File Selector Widget
    file_path = FileSelector(
        my_id=my_id_file_selector,
        default_directory=(
            export_path if os.path.isfile(export_path) else default_directory
        ),
    )

    # - Data Table Button Clicked
    if (
        my_id_plotly_data_columns_button in state
        and state[my_id_plotly_data_columns_button]
    ):
        state[my_id_data_representation] = "plot"

    # - Data Charts Button Clicked
    if my_id_show_data_table_button in state and state[my_id_show_data_table_button]:
        state[my_id_data_representation] = "show"

    # - Export Data Button Clicked
    if (
        my_id_column_data_exporter_button in state
        and state[my_id_column_data_exporter_button]
    ):
        selected_columns = state[my_id_selected_columns]
        if selected_columns[0] != None and selected_columns[1] != None:
            # Load and store data
            dl = DataLoader(file_path=file_path)
            dl.fetch()
            dl.store(export_path, *selected_columns)
            dl.clear()

            # Update state
            state[my_id_column_data_exporter] = True
        else:
            st.error("❌ &nbsp; Please select exactly two columns to export.")

    # If a file is selected
    if os.path.isfile(file_path):
        if state[my_id_column_data_exporter]:
            # Load data
            dl = DataLoader(file_path=export_path)
            dl.fetch()
            first_var_data, second_var_data = dl(0, 1)
            dl.clear()

            # Display success message
            st.success(
                f"✅ &nbsp; Successfully exported selected columns to:\n\n '{export_path}'",
            )

            # Describe data
            Label(
                "Data Description",
                key="data_description_label",
            )
            LatexRow(
                [
                    r"\#" + first_var_label + " = " + str(len(first_var_data)),
                    first_var_label
                    + " = "
                    + f"{min(first_var_data):.2e}"
                    + r"\, \ldots \,"
                    + f"{max(first_var_data):.2e}",
                ]
            )
            LatexRow(
                [
                    r"\#" + second_var_label + " = " + str(len(second_var_data)),
                    second_var_label
                    + " = "
                    + f"{min(second_var_data):.2e}"
                    + r" \, \ldots \, "
                    + f"{max(second_var_data):.2e}",
                ]
            )
            st.write("<br/>", unsafe_allow_html=True)

        else:
            # Load data
            dl = DataLoader(file_path=file_path)
            dl.fetch()
            data = dl.data
            dl.clear()

            # Display success message
            st.success("✅ &nbsp; Successfully fetched file.")

            # Select columns
            state[my_id_selected_columns] = ColumnDataSelector(
                my_id=my_id_column_data_exporter,
                first_var_label=first_var_label,
                second_var_label=second_var_label,
                data=data,
            )

        HR()

        # Buttons UI
        cols = st.columns([1, 1, 1, 1])
        with cols[0]:
            st.button(
                label="↩ &nbsp; Unselect File",
                key=my_id_unselect_file_button,
                use_container_width=True,
            )

        with cols[1]:
            st.button(
                label="📊 &nbsp; Data Charts",
                key=my_id_plotly_data_columns_button,
                use_container_width=True,
            )

        with cols[2]:
            st.button(
                label="📋 &nbsp; Data Table",
                key=my_id_show_data_table_button,
                use_container_width=True,
            )

        if not state[my_id_column_data_exporter]:
            with cols[3]:
                st.button(
                    label="📤 &nbsp; Export Data",
                    key=my_id_column_data_exporter_button,
                    use_container_width=True,
                )

        # Display Data Representation
        if state[my_id_data_representation] == "plot":
            PlotlyDataColumns(
                my_id=my_id_plotly_data_columns,
                file_path=file_path,
            )

        if state[my_id_data_representation] == "show":
            ShowDataTable(
                my_id=my_id_show_data_table,
                file_path=file_path,
            )

    return state[my_id_column_data_exporter]
