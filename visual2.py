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
import plotly.express as px

nltk.download('stopwords')
nltk.download('wordnet')

warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

st.set_page_config(page_title="Task Dashboard", layout="wide")

color_palette = sns.color_palette("autumn", as_cmap=True)

@st.cache_data
def load_data():
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the repository.")
        return pd.DataFrame()

    dataframes = []
    for filename in csv_files:
        df = pd.read_csv(filename)
        numeric_id = filename.split('-')[2] if '-' in filename else 'Unknown'
        df['ProjectID'] = numeric_id
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df['ProjectID-ID'] = combined_df['ProjectID'].astype(str) + "-" + combined_df['id'].astype(str)
    combined_df['Full_Name'] = combined_df['user_first_name'].astype(str) + " " + combined_df['user_last_name']
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
    combined_df["week"] = pd.to_datetime(combined_df["started_at"], errors="coerce").dt.isocalendar().week
    combined_df["month"] = pd.to_datetime(combined_df["started_at"], errors="coerce").dt.month
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

    combined_df['Categorized'] = combined_df['task_wo_punct_split_wo_stopwords_lemmatized'].apply(
        lambda x: categorize_words(x, categories))

    return combined_df

combined_df = load_data()

st.sidebar.header("Filters")
categories = st.sidebar.multiselect("Select Categories", options=combined_df['Categorized'].explode().unique())
date_filter = st.sidebar.date_input("Filter by Date", [])
search_term = st.sidebar.text_input("Search Task", "")
full_name_filter = st.sidebar.multiselect("Filter by Full Name", options=combined_df['Full_Name'].unique())

filtered_data = combined_df.copy()
if categories:
    def filter_categories(category_list):
        return any(cat in category_list for cat in categories)
    filtered_data = filtered_data[filtered_data['Categorized'].apply(filter_categories)]

if len(date_filter) == 2:
    filtered_data["started_at"] = pd.to_datetime(filtered_data["started_at"], errors="coerce").dt.tz_localize(None)
    start_date, end_date = pd.to_datetime(date_filter[0]), pd.to_datetime(date_filter[1])
    filtered_data = filtered_data[(filtered_data["started_at"] >= start_date) & (filtered_data["started_at"] <= end_date)]

if search_term:
    filtered_data = filtered_data[filtered_data['task'].str.contains(search_term, case=False, na=False)]

if full_name_filter:
    filtered_data = filtered_data[filtered_data['Full_Name'].isin(full_name_filter)]

csv_data = filtered_data.to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    label="\U0001F4E5 Download Filtered CSV",
    data=csv_data,
    file_name="filtered_data.csv",
    mime="text/csv"
)

# --- New Visualization Tab: Task Category Distribution ---
st.header("Task Category Distribution")

category_counts = Counter([cat for sublist in filtered_data['Categorized'] for cat in sublist])
category_df = pd.DataFrame(category_counts.items(), columns=["Category", "Count"])
category_df = category_df.sort_values("Count", ascending=False)

fig_cat = px.pie(
    category_df,
    values='Count',
    names='Category',
    title='Distribution of Task Categories',
    color_discrete_sequence=px.colors.sequential.Oranges
)
st.plotly_chart(fig_cat, use_container_width=True)
