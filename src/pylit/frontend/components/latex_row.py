import streamlit as st
from typing import List
from pylit.global_settings import FLOAT_DTYPE, INT_DTYPE


def LatexRow(elements: List[str]):
    ls = ""
    for element in elements:
        ls += "l "
    latex_row = r"\begin{array}{" + ls + "}"
    for i, element in enumerate(elements):
        if i == len(elements) - 1:
            break
        latex_row += element + r"\ &"
    latex_row += elements[-1]
    latex_row += r"""\end{array}"""
    st.latex(latex_row)
