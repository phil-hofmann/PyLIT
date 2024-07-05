import streamlit as st
from pylit.core import DataLoader

def show_data_table(my_id:str, file_path: str):
    dl = DataLoader(file_path=file_path)
    dl.fetch()
    st.table(dl.data)
    return