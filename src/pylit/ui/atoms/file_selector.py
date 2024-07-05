import os
import streamlit as st
from pylit import ui


def file_selector(my_id: str, default_directory: str):

    my_id_path = f"{my_id}_path"
    my_id_btn = f"{my_id}_btn"
    my_id_gobackbtn = f"{my_id}_gobackbtn"

    if my_id not in st.session_state:
        st.session_state[my_id] = default_directory

    def file_btn_callback(name: str):
        st.session_state[my_id] = os.path.join(st.session_state[my_id], name)

    def goback_btn_callback():
        st.session_state[my_id] = os.path.dirname(st.session_state[my_id])

    def selected_path_callback():
        new_path = st.session_state[my_id_path]
        if os.path.exists(new_path):
            st.session_state[my_id] = new_path
        else:
            st.error("Path does not exist.")

    col1, col2 = st.columns([6, 94])

    if not os.path.isfile(st.session_state[my_id]):
        # Go back button
        with col1:
            st.markdown("<div style='margin-top:30px'/>", unsafe_allow_html=True)
            st.button(f"↩", key=my_id_gobackbtn, on_click=goback_btn_callback)

        # Selected path input
        with col2:
            st.text_input(
                label="Selected Path",
                key=my_id_path,
                value=st.session_state[my_id],
                on_change=selected_path_callback,
            )

        st.markdown("<hr style='margin:0;padding:0;'/>", unsafe_allow_html=True)

    if os.path.isdir(st.session_state[my_id]):
        # Add custom CSS
        st.markdown(
            """
            <style>
            .button-after {
                display: none;
            }
            .element-container:has(.button-after) {
                display: none;
            }
            .element-container:has(.button-after) + div button {
                border: none;
                width: 100%;
                border-radius:0;
                text-align: left;
                margin:0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # List files and folders in the selected directory
        _, col2 = st.columns([6, 94])

        with col2:
            entries = [entry for entry in os.scandir(st.session_state[my_id])]

            # Sort entries in lexicographic order by their names
            entries.sort(key=lambda entry: entry.name)

            for i, entry in enumerate(entries):
                st.markdown(
                    '<span class="button-after"></span>', unsafe_allow_html=True
                )
                if entry.is_dir():
                    st.button(
                        f"📁 {entry.name}",
                        key=f"{my_id_btn}_{i}",
                        on_click=lambda x=entry.name: file_btn_callback(x),
                    )
                elif entry.is_file() and ui.utils.is_data_file(entry.name):
                    st.button(
                        f"📄 {entry.name}",
                        key=f"{my_id_btn}_{i}",
                        on_click=lambda x=entry.name: file_btn_callback(x),
                    )

    return st.session_state[my_id]


if __name__ == "__main__":
    file_selector(my_id="test", default_directory="/")
