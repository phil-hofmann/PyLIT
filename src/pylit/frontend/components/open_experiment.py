import os
import streamlit as st
from pylit.frontend.components.atoms import FileSelector


def OpenExperiment(my_id: str, workspace: str):
    st.write("Open Experiment")

    full_path = FileSelector(
        my_id,
        default_directory=st.session_state["workspace"],
        navigation_bar=False,
    )

    if not os.path.isdir(full_path):
        return None

    parent_dir = os.path.dirname(full_path)
    if os.path.samefile(parent_dir, workspace):
        return os.path.basename(full_path)

    return None
