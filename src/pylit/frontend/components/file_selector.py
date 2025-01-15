import os
import streamlit as st
from pylit.frontend.utils import is_data_file


def FileSelector(my_id: str, default_directory: str, navigation_bar: bool = True):

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

    # File Selector UI
    col = st.columns([5, 95], vertical_alignment="center")

    if not os.path.isfile(st.session_state[my_id]) and navigation_bar:
        # Go back button
        with col[0]:
            st.markdown(
                """
                <style>
                .back-button {
                    display: none;
                }
                .element-container:has(.back-button) + div button {
                    display: block;
                    width: 100%;
                    margin: 0;
                    margin-top: -25px;
                    padding: 0;
                    border: none;
                    border-radius: 0;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<span class="back-button"></span>', unsafe_allow_html=True)
            st.button(
                label="↩",
                key=my_id_gobackbtn,
                use_container_width=True,
                on_click=goback_btn_callback,
            )

        # Selected path input
        with col[1]:
            st.text_input(
                label="Selected Path",
                label_visibility="collapsed",
                key=my_id_path,
                value=st.session_state[my_id],
                on_change=selected_path_callback,
            )

    if os.path.isdir(st.session_state[my_id]):
        # Add custom CSS
        st.markdown(
            """
           <style>
            .folder-files-button {
                display: none;
            }
            .element-container:has(.folder-files-button) {
                display: none;
                margin: 0;
            }
            .element-container:has(.folder-files-button) + div {
                margin: -7px 0 -7px 0;
                padding: 0;
            }
            .element-container:has(.folder-files-button) + div button {
                display: block;
                width: 100%;
                margin: 0;
                border: none;
                border-radius: 0;
                text-align: left;
                border-bottom: solid 1.5px #f0f2f6;
            }
            .element-container:has(.folder-files-button) + div button p {
                padding: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # List files and folders in the selected directory
        entries = [entry for entry in os.scandir(st.session_state[my_id])]

        # Sort entries in lexicographic order by their names
        entries.sort(key=lambda entry: entry.name)

        for i, entry in enumerate(entries):
            st.markdown('<span class="folder-files-button"></span>', unsafe_allow_html=True)
            if entry.is_dir():
                st.button(
                    f"📁 &nbsp; {entry.name}",
                    key=f"{my_id_btn}_{i}",
                    on_click=lambda x=entry.name: file_btn_callback(x),
                )
            elif entry.is_file() and is_data_file(entry.name):
                st.button(
                    f"📄 &nbsp; {entry.name}",
                    key=f"{my_id_btn}_{i}",
                    on_click=lambda x=entry.name: file_btn_callback(x),
                )

    return st.session_state[my_id]
