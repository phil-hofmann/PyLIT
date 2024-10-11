import streamlit as st


def Label(text: str, key: str = "", mb: int = 0):
    # Type Conversion
    text = str(text)
    key = str(key)
    mb = str(int(mb))

    # Generate key if not provided
    if key == "":
        key = "label" + text.lower().replace(" ", "-")

    # Use HTML to apply the custom CSS class
    st.markdown(
        """
        <style>
        ."""
        + key
        + """-custom-markdown {
            margin-bottom: """
        + mb
        + """px; 
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Use HTML to apply the custom CSS class
    st.markdown(
        '<div class="'
        + key
        + '-custom-markdown"><strong>'
        + text
        + "</strong></div>",
        unsafe_allow_html=True,
    )
