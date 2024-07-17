import streamlit as st
from pylit.backend.core import DataLoader

def ShowDataTable(my_id:str, file_path: str):
    dl = DataLoader(file_path=file_path)
    dl.fetch()
    st.table(dl.data)
    return