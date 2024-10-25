import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from st_files_connection import FilesConnection

# Function to load data from GCS using @st.cache_resource
@st.cache_resource
def load_data_from_gcs():
    """
    Establish connection and load data from GCS.
    """
    conn = st.connection('gcs', type=FilesConnection)
    df = conn.read("apollo_streamlit/N1min.csv", input_format="csv")
    return df

# Functions for data cleansing methods
def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def filter_by_range(df, column, min_val, max_val):
    return df[(df[column] >= min_val) & (df[column] <= max_val)]

def delete_rows_by_datetime_range(df, column, start_datetime, end_datetime):
    df[column] = pd.to_datetime(df[column], errors='coerce')
    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)
    mask = (df[column] >= start_datetime) & (df[column] <= end_datetime)
    return df[~mask].reset_index(drop=True)

def display_logs():
    if st.session_state.logs:
        logs_df = pd.DataFrame(st.session_state.logs, columns=['Logging'])
        logs_df = logs_df.iloc[::-1].reset_index(drop=True)
        logs_df.index += 1
        st.table(logs_df)
        if st.button("Delete Logging"):
            st.session_state.logs = []
            st.success("Logging entries have been deleted.")
            st.rerun()
    else:
        st.write("No changes have been logged yet.")

def plot_original_and_processed_data(original_dataframe, processed_dataframe, datetime_column, cleansing_column):
    trace1 = go.Scatter(
        x=original_dataframe[datetime_column],
        y=original_dataframe[cleansing_column],
        mode='lines+markers',
        name='Original Data',
        line=dict(color='blue'),
        marker=dict(size=6)
    )
    trace2 = go.Scatter(
        x=processed_dataframe[datetime_column],
        y=processed_dataframe[cleansing_column],
        mode='lines+markers',
        name='Processed Data',
        line=dict(color='green'),
        marker=dict(size=6)
    )
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('Original Data', 'Processed Data'))
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=2, col=1)
    fig.update_layout(height=600, width=800, title_text="Subplots of Original and Processed Data")
    fig.update_yaxes(title_text="Values", row=1, col=1)
    fig.update_yaxes(title_text="Values", row=2, col=1)
    fig.update_xaxes(title_text="Date and Time", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

def save_to_history(df):
    st.session_state.history.append(df.copy())
    st.session_state.redo_stack.clear()

def undo():
    if len(st.session_state.history) > 1:
        st.session_state.redo_stack.append(st.session_state.processed_dataframe.copy())
        st.session_state.history.pop()
        st.session_state.processed_dataframe = st.session_state.history[-1]
        st.session_state.logs.append(f"Undo the last operation")
    else:
        st.session_state.processed_dataframe = st.session_state.history[0]

def redo():
    if st.session_state.redo_stack:
        st.session_state.history.append(st.session_state.processed_dataframe.copy())
        st.session_state.processed_dataframe = st.session_state.redo_stack.pop()
        st.session_state.logs.append(f"Redo the last operation")

def reset():
    st.session_state.processed_dataframe = st.session_state.original_dataframe.copy()
    st.session_state.history.clear()
    st.session_state.redo_stack.clear()
    save_to_history(st.session_state.original_dataframe)
    st.session_state.logs.append(f"Reset the dataframe to the original state")

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []  # Track past versions for undo
if 'redo_stack' not in st.session_state:
    st.session_state.redo_stack = []  # Track versions for redo
if 'processed_dataframe' not in st.session_state:
    st.session_state.processed_dataframe = None  # Store the processed dataframe
if 'logs' not in st.session_state:
    st.session_state.logs = []  # Store the logs

# Load original data using @st.cache_resource
if 'original_dataframe' not in st.session_state:
    st.session_state.original_dataframe = load_data_from_gcs()
    st.session_state.processed_dataframe = st.session_state.original_dataframe.copy()
    save_to_history(st.session_state.original_dataframe)

# Set up the Streamlit app layout
# st.set_page_config(layout="wide")
st.sidebar.title("Navigation")

if "page" not in st.session_state:
    st.session_state.page = "Visualization"
page = st.sidebar.radio("Go to", ["Visualization", "Correlation plot", "Export Processed File"], index=0)
with st.sidebar:
    display_logs()

if page == "Visualization":
    st.title("Visualization Page")
    
    datetime_column = st.selectbox("Select the 'DATETIME' column for the x-axis:", st.session_state.original_dataframe.columns)
    cleansing_column = st.selectbox("Select the column to cleanse or filter:", st.session_state.original_dataframe.columns)
    
    plot_original_and_processed_data(st.session_state.original_dataframe, st.session_state.processed_dataframe, datetime_column, cleansing_column)

    col1, col2, col3, _ = st.columns([1, 1, 1, 15])
    with col1:
        if st.button("Undo", disabled=len(st.session_state.history) == 1):
            undo()
            st.rerun()

    with col2:
        if st.button("Redo", disabled=len(st.session_state.redo_stack) == 0):
            redo()
            st.rerun()

    with col3:
        if st.button("Reset", disabled=len(st.session_state.history) == 1):
            reset()
            st.rerun()

elif page == "Correlation plot":
    st.title("Correlation plot")
    x_col = st.selectbox("Select X column", st.session_state.processed_dataframe.columns)
    y_col = st.selectbox("Select Y column", st.session_state.processed_dataframe.columns)
    fig = px.scatter(
        x=st.session_state.processed_dataframe[x_col],
        y=st.session_state.processed_dataframe[y_col],
        labels={'x': x_col, 'y': y_col},
        title=f'Scatter plot of {x_col} vs {y_col}'
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Export Processed File":
    st.title("Export Processed File")
    buffer = BytesIO()
    st.session_state.processed_dataframe.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="Download Processed CSV",
        data=buffer,
        file_name="processed_file.csv",
        mime="text/csv"
    )
