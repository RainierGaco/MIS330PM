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
import plotly.express as px
import plotly.graph_objects as go

nltk.download('stopwords')
nltk.download('wordnet')

warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

st.set_page_config(page_title="Task Dashboard", layout="wide")

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
        st.warning("No CSV files found in the repository.")
        return pd.DataFrame()

    dataframes = []
    for filename in csv_files:
        df = pd.read_csv(filename)
        numeric_id = filename.split('-')[2] if '-' in filename else 'Unknown'
        df['ProjectID'] = numeric_id
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df['ProjectID-ID'] = combined_df['ProjectID'].astype(str) + "-" + combined_df['id'].astype(str)
    combined_df['Full_Name'] = combined_df['user_first_name'].astype(str) + " " + combined_df['user_last_name'].astype(str)
    combined_df['week'] = pd.to_datetime(combined_df['started_at'], errors="coerce").dt.isocalendar().week
    combined_df['month'] = pd.to_datetime(combined_df['started_at'], errors="coerce").dt.month
    combined_df['year'] = pd.to_datetime(combined_df['started_at'], errors="coerce").dt.year

    combined_df['task_wo_punct'] = combined_df['task'].apply(lambda x: ''.join([char for char in str(x) if char not in string.punctuation]))
    combined_df['task_wo_punct_split'] = combined_df['task_wo_punct'].apply(lambda x: re.split(r'\W+', str(x).lower()))

    stopword = nltk.corpus.stopwords.words('english')
    combined_df['task_wo_punct_split_wo_stopwords'] = combined_df['task_wo_punct_split'].apply(
        lambda x: [word for word in x if word not in stopword]
    )

    lemmatizer = WordNetLemmatizer()
    combined_df['task_wo_punct_split_wo_stopwords_lemmatized'] = combined_df['task_wo_punct_split_wo_stopwords'].apply(
        lambda x: [lemmatizer.lemmatize(word) for word in x]
    )

    combined_df["Hours"] = combined_df["minutes"] / 60
    combined_df["year_month"] = pd.to_datetime(combined_df["started_at"], errors="coerce").dt.to_period("M")

    categories = {
        "technology": ["website", "sql", "backend", "repository", "ai", "coding", "file", "database", "application", "program", "flask", "html", "css", "javascript"],
        "actions": ["reviewed", "created", "tested", "fixed", "debugged", "implemented", "researched", "planned", "updated", "designed", "documented", "analyzed", "optimized", "added", "removed"],
        "design": ["logo", "design", "styling", "layout", "responsive", "theme", "navbar", "icon", "image", "photo", "redesigning", "wireframes"],
        "writing": ["blog", "guide", "documentation", "report", "note", "summary", "draft", "content", "copywriting"],
        "meetings": ["meeting", "call", "discussion", "session", "presentation", "team"],
        "business": ["grant", "funding", "startup", "loan", "entrepreneur", "business", "government"],
        "errors": ["bug", "error", "issue", "fixing", "debugging", "problem", "mistake"],
        "time": ["hour", "day", "week", "month", "year"],
        "miscellaneous": []
    }

    def categorize_words(words, categories):
        matched_categories = set()
        for word in words:
            found = False
            for category, keywords in categories.items():
                if word in keywords:
                    matched_categories.add(category)
                    found = True
                    break
            if not found:
                matched_categories.add("miscellaneous")
        return list(matched_categories)

    combined_df['Categorized'] = combined_df['task_wo_punct_split_wo_stopwords_lemmatized'].apply(lambda x: categorize_words(x, categories))

    return combined_df

combined_df = load_data()

st.sidebar.header("Filters")
categories = st.sidebar.multiselect("Select Categories", options=combined_df['Categorized'].explode().unique())
date_filter = st.sidebar.date_input("Filter by Date", [])
full_name_filter = st.sidebar.multiselect("Filter by Full Name", options=combined_df['Full_Name'].unique())

filtered_df = combined_df.copy()

if categories:
    def filter_categories(category_list):
        return any(cat in category_list for cat in categories)
    filtered_df = filtered_df[filtered_df['Categorized'].apply(filter_categories)]

if len(date_filter) == 2:
    filtered_df["started_at"] = pd.to_datetime(filtered_df["started_at"], errors="coerce").dt.tz_localize(None)
    start_date = pd.to_datetime(date_filter[0])
    end_date = pd.to_datetime(date_filter[1])
    filtered_df = filtered_df[(filtered_df["started_at"] >= start_date) & (filtered_df["started_at"] <= end_date)]

if full_name_filter:
    filtered_df = filtered_df[filtered_df['Full_Name'].isin(full_name_filter)]

st.sidebar.download_button(
    label="⬇️ Download Filtered CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name="filtered_data.csv",
    mime="text/csv"
)

st.title("📊 Task Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tasks", filtered_df.shape[0])
col2.metric("Total Hours", round(filtered_df["Hours"].sum(), 2))
col3.metric("Unique Users", filtered_df["Full_Name"].nunique())
col4.metric("Unique Projects", filtered_df["ProjectID"].nunique())

# TABS FOR VISUALIZATION
category_tab, time_tab, wordcloud_tab, table_tab = st.tabs(["📂 Category Breakdown", "📈 Time Series", "☁️ Word Cloud", "📋 Data Table"])

with category_tab:
    st.subheader("Tasks by Category")
    cat_counts = filtered_df.explode('Categorized')['Categorized'].value_counts()
    bar_fig = px.bar(cat_counts, x=cat_counts.index, y=cat_counts.values, title="Task Counts by Category", color=cat_counts.values, color_continuous_scale='greens')
    st.plotly_chart(bar_fig, use_container_width=True)

    pie_fig = px.pie(cat_counts, names=cat_counts.index, values=cat_counts.values, title="Category Distribution")
    st.plotly_chart(pie_fig, use_container_width=True)

with time_tab:
    st.subheader("Hours Worked Over Time")
    time_df = filtered_df.groupby('year_month')['Hours'].sum().reset_index()
    time_df['year_month'] = time_df['year_month'].astype(str)
    line_fig = px.line(time_df, x="year_month", y="Hours", title="Total Hours per Month", markers=True)
    st.plotly_chart(line_fig, use_container_width=True)

    task_time = filtered_df.groupby('year_month')['task'].count().reset_index(name="Task Count")
    task_time['year_month'] = task_time['year_month'].astype(str)
    area_fig = px.area(task_time, x="year_month", y="Task Count", title="Tasks Over Time")
    st.plotly_chart(area_fig, use_container_width=True)

with wordcloud_tab:
    st.subheader("Top 50 Words")
    all_words = [word for sublist in filtered_df['task_wo_punct_split_wo_stopwords_lemmatized'] for word in sublist]
    word_counts = Counter(all_words).most_common(50)

    if word_counts:
        # Generate word cloud
        wordcloud = WordCloud(width=1000, height=500, background_color='black', colormap='Greens').generate_from_frequencies(dict(word_counts))
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)

        # Prepare dataframe for visualization
        word_df = pd.DataFrame(word_counts, columns=["Word", "Count"])

        # Use Plotly Treemap for Top 20 Words instead of bar chart
        top_words_df = word_df.head(20)
        treemap_fig = px.treemap(
            top_words_df,
            path=['Word'],
            values='Count',
            color='Count',
            color_continuous_scale='Greens',
            title="Top 20 Words Treemap"
        )
        st.plotly_chart(treemap_fig, use_container_width=True)

    else:
        st.info("No word frequency data available for the selected filters.")

with table_tab:
    st.subheader("Filtered Data Table")
    st.dataframe(filtered_df, use_container_width=True)

    top_users = filtered_df.groupby('Full_Name')['Hours'].sum().reset_index().sort_values(by="Hours", ascending=False)
    user_fig = px.bar(top_users, x="Full_Name", y="Hours", title="Hours Worked by User", color="Hours", color_continuous_scale="greens")
    st.plotly_chart(user_fig, use_container_width=True)
