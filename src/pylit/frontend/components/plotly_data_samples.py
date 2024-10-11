import streamlit as st
import plotly.graph_objects as go

from pylit.global_settings import ARRAY


def PlotlyDataSamples(data: ARRAY, color: str) -> None:

    num_samples = len(data)
    tabs = st.tabs([f"{i+1}" for i in range(num_samples)])

    for i in range(num_samples):
        # Extract the i-th sample from the data
        sample = data[i]

        # Create a Plotly figure
        fig = go.Figure()

        # Add the sample to the figure as a line plot
        fig.add_trace(
            go.Scatter(
                y=sample,
                mode="lines",
                line=dict(color=color),
            )
        )

        # Update layout for the figure
        fig.update_layout(
            xaxis_title="Index",
            yaxis_title="Value",
        )

        # Display the figure
        with tabs[i]:
            st.plotly_chart(fig)
