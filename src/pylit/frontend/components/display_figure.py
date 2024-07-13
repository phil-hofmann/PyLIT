import streamlit as st
from pylit.frontend.utils import is_plotly_figure


def DisplayFigure(fig):
    if isinstance(fig, list) and all(is_plotly_figure(fig) for fig in fig):
        num_figures = len(fig)
        num_rows = (num_figures + 1) // 2  # Add 1 to handle odd number of figures

        # Create subplots
        for i in range(num_rows):
            lb, ub = 2 * i, 2 * i + 1
            cols = st.columns(2)
            with cols[0]:
                st.plotly_chart(fig[lb], use_container_width=True)
            if ub < num_figures:
                with cols[1]:
                    st.plotly_chart(fig[ub], use_container_width=True)
    else:
        st.pyplot(fig)
