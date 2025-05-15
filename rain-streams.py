import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Task Dashboard", layout="wide")

st.markdown("""
    <style>
        .main, .block-container {
            background-color: red;
            color: white;
        }
        .css-18e3th9 {
            background-color: red;
        }
        .css-1d391kg, .css-1v0mbdj, .css-ffhzg2, .css-1dp5vir, .stMetric {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Task Dashboard (Red Theme)")

# Load data
@st.cache_data
def load_data():
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]
    if not csv_files:
        st.warning("No CSV files found in the repository.")
        return pd.DataFrame()

    combined_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
    combined_df["Hours"] = combined_df["minutes"] / 60
    return combined_df

combined_df = load_data()

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Tasks", combined_df.shape[0])
col2.metric("Total Hours", round(combined_df["Hours"].sum(), 2))
col3.metric("Unique Users", combined_df["user_first_name"].nunique())

# Visualization Tabs
tab1, tab2, tab3 = st.tabs(["📋 Data Table", "📊 User Analysis", "📅 Time Analysis"])

with tab1:
    st.subheader("Filtered Data Table")
    st.dataframe(combined_df, use_container_width=True)

with tab2:
    st.subheader("Tasks by User")
    user_task_counts = combined_df["user_first_name"].value_counts()
    bar_fig = px.bar(user_task_counts, x=user_task_counts.index, y=user_task_counts.values, title="Tasks per User", color=user_task_counts.values, color_continuous_scale='reds')
    st.plotly_chart(bar_fig, use_container_width=True)

with tab3:
    st.subheader("Hours Worked Over Time")
    time_df = combined_df.groupby("started_at")['Hours'].sum().reset_index()
    line_fig = px.line(time_df, x="started_at", y="Hours", title="Hours Over Time", markers=True, color_discrete_sequence=['red'])
    st.plotly_chart(line_fig, use_container_width=True)
