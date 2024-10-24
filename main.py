# streamlit_app.py

import streamlit as st
from st_files_connection import FilesConnection

# Define a function to load data and cache it using @st.cache_data
@st.cache_data(ttl=6000)
def load_data():
    # Create connection object and retrieve file contents
    conn = st.connection('gcs', type=FilesConnection)
    df = conn.read("apollo_streamlit/N1min.csv", input_format="csv")
    return df

# Load the data
df = load_data()

# Display the dataframe
st.write(df)

