import pandas as pd
import streamlit as st

def DictionaryTable(dictionary: dict):
    # Create a DataFrame and transpose it
    df = pd.DataFrame(dictionary).T

    # Set the first row as the header and drop it
    df.columns = df.iloc[0]
    df = df[1:]

    # Display the DataFrame without the index
    st.dataframe(df)