import plotly
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pylit.backend.core import DataLoader


def PlotlyDataColumns(my_id: str, file_path: str):
    dl = DataLoader(file_path=file_path)
    dl.fetch()

    # Convert data to DataFrame
    df = pd.DataFrame(
        dl.data, columns=[f"Column {i+1}" for i in range(dl.data.shape[1])]
    )

    # Create traces for each column
    data = []
    for i, column in enumerate(df.columns):
        trace = go.Scatter(
            x=df.index,
            y=df[column],
            mode="lines",
            name=column,
            line=dict(color=plotly.colors.qualitative.Plotly[i]),
        )
        data.append(trace)

    # Create layout
    layout = go.Layout(
        title="Plot of Columns",
        xaxis=dict(title="Index"),
        yaxis=dict(title="Value"),
    )

    # Create figure
    fig = go.Figure(data=data, layout=layout)

    # Display the plotly chart
    st.plotly_chart(fig)
