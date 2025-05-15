import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

nltk.download('stopwords')
nltk.download('wordnet')

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Task Dashboard", layout="wide")

# Custom styling
st.markdown("""
    <style>
        .main, .block-container {
            background-color: black;
            color: lightgreen;
        }
        .css-18e3th9 {
            background-color: black;
        }
        .css-1d391kg, .css-1v0mbdj, .css-ffhzg2, .css-1dp5vir, .stMetric {
            color: lightgreen !important;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]
    if not csv_files:
        st.warning("No CSV files found.")
        return pd.DataFrame()

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['ProjectID'] = file.split('-')[2] if '-' in file else 'Unknown'
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df['ProjectID-ID'] = df['ProjectID'].astype(str) + "-" + df['id'].astype(str)
    df['Full_Name'] = df['user_first_name'].astype(str) + " " + df['user_last_name'].astype(str)
    df['started_at'] = pd.to_datetime(df['started_at'], errors="coerce").dt.tz_localize(None)
    df['week'] = df['started_at'].dt.isocalendar().week
    df['month'] = df['started_at'].dt.month
    df['year_month'] = df['started_at'].dt.to_period("M")
    df["Hours"] = df["minutes"] / 60

    # Preprocessing tasks
    df['task_wo_punct'] = df['task'].apply(lambda x: ''.join(ch for ch in str(x) if ch not in string.punctuation))
    df['task_wo_punct_split'] = df['task_wo_punct'].apply(lambda x: re.split(r'\W+', x.lower()))
    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    df['task_wo_punct_split_wo_stopwords_lemmatized'] = df['task_wo_punct_split'].apply(
        lambda x: [lemmatizer.lemmatize(word) for word in x if word not in stopwords]
    )

    # Categorization
    categories = {
        "technology": ["website", "sql", "backend", "repository", "ai", "coding", "database", "application", "flask", "html", "css", "javascript"],
        "actions": ["reviewed", "created", "tested", "fixed", "implemented", "updated", "documented", "analyzed", "added", "removed"],
        "design": ["logo", "design", "layout", "theme", "navbar", "icon"],
        "writing": ["blog", "guide", "documentation", "note", "summary", "content"],
        "meetings": ["meeting", "call", "discussion", "presentation", "team"],
        "business": ["grant", "funding", "startup", "loan", "business"],
        "errors": ["bug", "error", "issue", "fixing", "debugging"],
        "miscellaneous": []
    }

    def categorize(words):
        matched = set()
        for word in words:
            for category, keywords in categories.items():
                if word in keywords:
                    matched.add(category)
                    break
            else:
                matched.add("miscellaneous")
        return list(matched)

    df['Categorized'] = df['task_wo_punct_split_wo_stopwords_lemmatized'].apply(lambda x: categorize(x))
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
selected_cats = st.sidebar.multiselect("Select Categories", df['Categorized'].explode().unique())
date_filter = st.sidebar.date_input("Filter by Date", [])
name_filter = st.sidebar.multiselect("Filter by Full Name", df['Full_Name'].unique())

filtered_df = df.copy()
if selected_cats:
    filtered_df = filtered_df[filtered_df['Categorized'].apply(lambda x: any(cat in x for cat in selected_cats))]
if len(date_filter) == 2:
    start, end = pd.to_datetime(date_filter[0]), pd.to_datetime(date_filter[1])
    filtered_df = filtered_df[(filtered_df['started_at'] >= start) & (filtered_df['started_at'] <= end)]
if name_filter:
    filtered_df = filtered_df[filtered_df['Full_Name'].isin(name_filter)]

# Tabs layout
tab1, tab2, tab3, tab4 = st.tabs(["📂 Category Breakdown", "📈 Time Series", "🔠 Word Cloud", "📋 Data Table"])

# Tab 1: Category Breakdown
with tab1:
    st.header("📂 Task Categories")
    cat_counts = filtered_df.explode('Categorized')['Categorized'].value_counts()
    fig1 = px.bar(cat_counts, x=cat_counts.index, y=cat_counts.values,
                  labels={"x": "Category", "y": "Task Count"},
                  color=cat_counts.values, color_continuous_scale="greens")
    st.plotly_chart(fig1, use_container_width=True)

    pie_fig = px.pie(cat_counts, names=cat_counts.index, values=cat_counts.values, title="Category Distribution")
    st.plotly_chart(pie_fig, use_container_width=True)

# Tab 2: Time Series
with tab2:
    st.header("📈 Time Series Overview")
    time_df = filtered_df.groupby('year_month')['Hours'].sum().reset_index()
    time_df['year_month'] = time_df['year_month'].astype(str)

    line_fig = px.line(time_df, x="year_month", y="Hours", markers=True, title="Total Hours Over Time")
    st.plotly_chart(line_fig, use_container_width=True)

    task_time = filtered_df.groupby('year_month').size().reset_index(name="Task Count")
    task_time['year_month'] = task_time['year_month'].astype(str)
    area_fig = px.area(task_time, x="year_month", y="Task Count", title="Tasks Over Time")
    st.plotly_chart(area_fig, use_container_width=True)

# Tab 3: Word Cloud
with tab3:
    st.header("🔠 Word Cloud")
    words = [word for row in filtered_df['task_wo_punct_split_wo_stopwords_lemmatized'] for word in row]
    word_counts = Counter(words).most_common(50)

    if word_counts:
        wordcloud = WordCloud(width=1000, height=500, background_color='black', colormap='Greens').generate_from_frequencies(dict(word_counts))
        fig_wc, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)

        word_df = pd.DataFrame(word_counts, columns=["Word", "Count"])
        bar_fig = px.bar(word_df.head(20), x="Word", y="Count", title="Top 20 Words")
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.write("No word data available.")

# Tab 4: Data Table
with tab4:
    st.header("📋 Filtered Task Table")
    st.dataframe(filtered_df, use_container_width=True)

    top_users = filtered_df.groupby('Full_Name')['Hours'].sum().reset_index().sort_values(by="Hours", ascending=False)
    user_fig = px.bar(top_users, x="Full_Name", y="Hours", title="Hours Worked by User", color="Hours", color_continuous_scale="greens")
    st.plotly_chart(user_fig, use_container_width=True)
