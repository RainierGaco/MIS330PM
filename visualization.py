import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import nltk
import plotly.express as px
import plotly.graph_objects as go
from nltk.stem import WordNetLemmatizer

# NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')

warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

# Streamlit config
st.set_page_config(page_title="Task Dashboard", layout="wide")

# Custom dark theme
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #0e1117;
    }
    .css-1v3fvcr, .css-18e3th9 {
        background-color: #0e1117;
    }
    div[data-baseweb="tab"] > div[role="tablist"] > button[role="tab"] {
        color: #FFFFFF;
    }
    div[data-baseweb="tab"] > div[role="tablist"] > button[role="tab"][aria-selected="true"] {
        color: #1E90FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
@st.cache_data
def load_data():
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        return pd.DataFrame()
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        project_id = file.split('-')[2] if '-' in file else 'Unknown'
        df['ProjectID'] = project_id
        dataframes.append(df)
    combined = pd.concat(dataframes, ignore_index=True)
    combined['ProjectID-ID'] = combined['ProjectID'].astype(str) + "-" + combined['id'].astype(str)
    combined['Full_Name'] = combined['user_first_name'].astype(str) + " " + combined['user_last_name'].astype(str)
    return combined

# Load and filter data
df = load_data()
st.sidebar.header("Filters")

if 'Categorized' in df.columns:
    categories = st.sidebar.multiselect("Select Categories", options=df['Categorized'].dropna().unique())
    if categories:
        df = df[df['Categorized'].isin(categories)]

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "⏱️ Hours", "🧠 Insights"])

# --- Tab 1: Overview ---
with tab1:
    st.subheader("📌 Preview of Filtered Data (First 100 Rows)")
    st.dataframe(df.head(100), use_container_width=True)

    # --- Bar Chart: Task Count by User ---
    st.subheader("👤 Task Count by User")
    user_task_counts = df['Full_Name'].value_counts().reset_index()
    user_task_counts.columns = ['Full_Name', 'Task Count']

    fig1 = px.bar(
        user_task_counts,
        x='Full_Name',
        y='Task Count',
        color='Task Count',
        color_continuous_scale='blues',
        title="Tasks per User"
    )
    fig1.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font_color='white',
        xaxis_title='User',
        yaxis_title='Number of Tasks',
        title_x=0.5
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Pie Chart: Task Distribution by Category ---
    if 'Categorized' in df.columns:
        st.subheader("📘 Task Distribution by Category")
        cat_counts = df['Categorized'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig2 = px.pie(cat_counts, names='Category', values='Count', title="Tasks per Category", hole=0.4)
        fig2.update_traces(textinfo='percent+label')
        fig2.update_layout(paper_bgcolor='#0e1117', font_color='white', title_x=0.5)
        st.plotly_chart(fig2, use_container_width=True)

# --- Tab 2: Hours ---
with tab2:
    if 'hours' in df.columns and 'Full_Name' in df.columns:
        st.subheader("⏱️ Total Hours per User")
        hours_df = df.groupby('Full_Name')['hours'].sum().reset_index()
        fig3 = px.bar(
            hours_df,
            x='Full_Name',
            y='hours',
            color='hours',
            color_continuous_scale='viridis',
            title='Total Logged Hours by User'
        )
        fig3.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='white',
            xaxis_title='User',
            yaxis_title='Total Hours',
            title_x=0.5
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No 'hours' column found in the dataset.")

# --- Tab 3: Insights ---
with tab3:
    st.subheader("🔍 Correlation Heatmap (Numerical Features)")
    if not df.select_dtypes(include='number').empty:
        corr = df.select_dtypes(include='number').corr()
        fig4 = px.imshow(corr, text_auto=True, color_continuous_scale='deep', title="Correlation Matrix")
        fig4.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='white',
            title_x=0.5
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Optional Sunburst for hierarchy
    if {'Categorized', 'Full_Name'}.issubset(df.columns):
        st.subheader("🌐 Task Hierarchy (User → Category)")
        sunburst_data = df.groupby(['Full_Name', 'Categorized']).size().reset_index(name='Count')
        fig5 = px.sunburst(sunburst_data, path=['Full_Name', 'Categorized'], values='Count',
                           color='Count', color_continuous_scale='plasma', title="User & Task Category Breakdown")
        fig5.update_layout(paper_bgcolor='#0e1117', font_color='white', title_x=0.5)
        st.plotly_chart(fig5, use_container_width=True)
