import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import os
import warnings
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
import plotly.express as px

# Download required NLTK data once
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

st.set_page_config(page_title="Task Dashboard", layout="wide")

@st.cache_data
def load_data():
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]
    if not csv_files:
        st.warning("No CSV files found in the current directory.")
        return pd.DataFrame()

    dataframes = []
    for filename in csv_files:
        try:
            df = pd.read_csv(filename)
            # Attempt to extract numeric ID safely
            parts = filename.split('-')
            numeric_id = parts[2] if len(parts) > 2 else 'Unknown'
            df['ProjectID'] = numeric_id
            dataframes.append(df)
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")

    if not dataframes:
        return pd.DataFrame()

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Ensure necessary columns exist
    required_cols = ['id', 'user_first_name', 'user_last_name', 'started_at', 'task', 'minutes']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        st.error(f"Missing columns in data: {missing_cols}")
        return pd.DataFrame()

    combined_df['ProjectID-ID'] = combined_df['ProjectID'].astype(str) + "-" + combined_df['id'].astype(str)
    combined_df['Full_Name'] = combined_df['user_first_name'].astype(str) + " " + combined_df['user_last_name'].astype(str)

    # Convert dates
    combined_df['started_at'] = pd.to_datetime(combined_df['started_at'], errors='coerce')
    combined_df['week'] = combined_df['started_at'].dt.isocalendar().week
    combined_df['month'] = combined_df['started_at'].dt.month
    combined_df['year'] = combined_df['started_at'].dt.year
    combined_df['year_month'] = combined_df['started_at'].dt.to_period('M')

    # Clean and process task text
    combined_df['task_wo_punct'] = combined_df['task'].astype(str).apply(
        lambda x: ''.join(ch for ch in x if ch not in string.punctuation)
    )
    combined_df['task_wo_punct_split'] = combined_df['task_wo_punct'].str.lower().str.split()

    stopwords = set(nltk.corpus.stopwords.words('english'))
    combined_df['task_wo_punct_split_wo_stopwords'] = combined_df['task_wo_punct_split'].apply(
        lambda words: [word for word in words if word not in stopwords]
    )

    lemmatizer = WordNetLemmatizer()
    combined_df['task_wo_punct_split_wo_stopwords_lemmatized'] = combined_df['task_wo_punct_split_wo_stopwords'].apply(
        lambda words: [lemmatizer.lemmatize(word) for word in words]
    )

    combined_df["Hours"] = combined_df["minutes"] / 60

    # Define categories
    categories = {
        "technology": ["website", "sql", "backend", "repository", "ai", "coding", "file", "database", "application",
                       "program", "flask", "html", "css", "javascript"],
        "actions": ["reviewed", "created", "tested", "fixed", "debugged", "implemented", "researched", "planned",
                    "updated", "designed", "documented", "analyzed", "optimized", "added", "removed"],
        "design": ["logo", "design", "styling", "layout", "responsive", "theme", "navbar", "icon", "image", "photo",
                   "redesigning", "wireframes"],
        "writing": ["blog", "guide", "documentation", "report", "note", "summary", "draft", "content", "copywriting"],
        "meetings": ["meeting", "call", "discussion", "session", "presentation", "team"],
        "business": ["grant", "funding", "startup", "loan", "entrepreneur", "business", "government"],
        "errors": ["bug", "error", "issue", "fixing", "debugging", "problem", "mistake"],
        "time": ["hour", "day", "week", "month", "year"],
        "miscellaneous": []
    }

    def categorize_words(words, categories):
        matched = set()
        for word in words:
            found = False
            for cat, keywords in categories.items():
                if word in keywords:
                    matched.add(cat)
                    found = True
                    break
            if not found:
                matched.add("miscellaneous")
        return list(matched)

    combined_df['Categorized'] = combined_df['task_wo_punct_split_wo_stopwords_lemmatized'].apply(
        lambda x: categorize_words(x, categories)
    )

    return combined_df


combined_df = load_data()

if combined_df.empty:
    st.info("No data available to display. Please add CSV files to the directory.")
else:
    st.sidebar.header("Filters")

    all_categories = combined_df['Categorized'].explode().unique()
    selected_categories = st.sidebar.multiselect("Select Categories", options=sorted(all_categories))

    date_filter = st.sidebar.date_input("Filter by Date", value=[])
    search_term = st.sidebar.text_input("Search Task", "")
    full_name_options = sorted(combined_df['Full_Name'].unique())
    selected_full_names = st.sidebar.multiselect("Filter by Full Name", options=full_name_options)

    filtered_data = combined_df.copy()

    if selected_categories:
        def category_filter(cat_list):
            return any(cat in cat_list for cat in selected_categories)
        filtered_data = filtered_data[filtered_data['Categorized'].apply(category_filter)]

    if len(date_filter) == 2:
        start_date, end_date = pd.to_datetime(date_filter[0]), pd.to_datetime(date_filter[1])
        filtered_data = filtered_data[
            (filtered_data['started_at'] >= start_date) & (filtered_data['started_at'] <= end_date)
        ]

    if search_term:
        filtered_data = filtered_data[filtered_data['task'].str.contains(search_term, case=False, na=False)]

    if selected_full_names:
        filtered_data = filtered_data[filtered_data['Full_Name'].isin(selected_full_names)]

    csv_data = filtered_data.to_csv(index=False).encode('utf-8')

    st.sidebar.download_button(
        label="📥 Download Filtered CSV",
        data=csv_data,
        file_name="filtered_data.csv",
        mime="text/csv"
    )

    # Visualization: Top 50 Most Common Lemmatized Words
    with st.expander("🔍 Top 50 Most Common Searches (Lemmatized)", expanded=True):
        all_words = [word for sublist in filtered_data['task_wo_punct_split_wo_stopwords_lemmatized'] for word in sublist]
        word_counts = Counter(all_words).most_common(50)
        if word_counts:
            words, counts = zip(*word_counts)
            df_plot = pd.DataFrame({'Word': words, 'Count': counts})
            fig = px.bar(
                df_plot,
                x='Word',
                y='Count',
                color='Count',
                color_continuous_scale='YlOrRd',
                title="Top 50 Most Common Lemmatized Words",
                labels={'Count': 'Frequency'},
                hover_data={'Word': True, 'Count': True}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(title="No Data Found for Search Frequency")
            fig.update_layout(
                xaxis={'visible': False},
                yaxis={'visible': False},
                annotations=[{
                    'text': "No search data available",
                    'xref': "paper",
                    'yref': "paper",
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            )
            st.plotly_chart(fig, use_container_width=True)
