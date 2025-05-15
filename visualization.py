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
import plotly.graph_objects as go
import plotly.express as px

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
        lambda x: [word for word in x if word not in stopword and word]
    )

    lemmatizer = WordNetLemmatizer()
    combined_df['task_wo_punct_split_wo_stopwords_lemmatized'] = combined_df['task_wo_punct_split_wo_stopwords'].apply(
        lambda x: [lemmatizer.lemmatize(word) for word in x]
    )

    combined_df['Lemmatized_Task'] = combined_df['task_wo_punct_split_wo_stopwords_lemmatized'].apply(
        lambda x: ' '.join(sorted(set(x))) if x else ''
    )

    # Highlight which words were changed during lemmatization
    def highlight_lemmatized(original, lemmatized):
        return [f"**{o} → {l}**" if o != l else o for o, l in zip(original, lemmatized)]

    combined_df['lemmatization_changes'] = combined_df.apply(
        lambda row: highlight_lemmatized(row['task_wo_punct_split_wo_stopwords'], row['task_wo_punct_split_wo_stopwords_lemmatized']),
        axis=1
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

    combined_df['Categorized'] = combined_df['task_wo_punct_split_wo_stopwords_lemmatized'].apply(lambda x: categorize_words(x, categories))

    return combined_df

combined_df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")
categories = st.sidebar.multiselect("Select Categories", options=combined_df['Categorized'].explode().unique())
date_filter = st.sidebar.date_input("Filter by Date", [])
full_name_filter = st.sidebar.multiselect("Filter by Full Name", options=combined_df['Full_Name'].unique())

# Added slider for rows to display
max_rows = st.sidebar.slider("Number of rows to display", min_value=10, max_value=100, value=50, step=10)

filtered_data = combined_df.copy()

if categories:
    def filter_categories(category_list):
        return any(cat in category_list for cat in categories)
    filtered_data = filtered_data[filtered_data['Categorized'].apply(filter_categories)]

if len(date_filter) == 2:
    filtered_data["started_at"] = pd.to_datetime(filtered_data["started_at"], errors="coerce").dt.tz_localize(None)
    start_date = pd.to_datetime(date_filter[0])
    end_date = pd.to_datetime(date_filter[1])
    filtered_data = filtered_data[
        (filtered_data["started_at"] >= start_date) &
        (filtered_data["started_at"] <= end_date)
    ]

if full_name_filter:
    filtered_data = filtered_data[filtered_data['Full_Name'].isin(full_name_filter)]

# Download button
csv_data = filtered_data.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="⬇️ Download Filtered CSV",
    data=csv_data,
    file_name="filtered_data.csv",
    mime="text/csv"
)

# --- OVERVIEW DASHBOARD ---

st.title("📊 Task Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Tasks", filtered_data.shape[0])
col2.metric("Total Hours", round(filtered_data["Hours"].sum(), 2))
col3.metric("Unique Users", filtered_data["Full_Name"].nunique())
col4.metric("Unique Projects", filtered_data["ProjectID"].nunique())

# Tasks by Category
cat_counts = filtered_data.explode('Categorized')['Categorized'].value_counts()

fig_cat = go.Figure()
fig_cat.add_trace(go.Bar(
    x=cat_counts.index,
    y=cat_counts.values,
    marker=dict(
        color=cat_counts.values,
        colorscale='Greens',
        line=dict(width=0.8, color='DarkGreen')
    ),
    hovertemplate='Category: %{x}<br>Tasks: %{y}<extra></extra>'
))

fig_cat.update_layout(
    title='Task Counts by Category',
    xaxis_title='Category',
    yaxis_title='Number of Tasks',
    xaxis_tickangle=-45,
    plot_bgcolor='black',
    font=dict(family='Arial', size=14, color='lightgreen'),
    margin=dict(l=40, r=40, t=70, b=100),
    yaxis=dict(
        gridcolor='darkgreen',
        zeroline=True,
        zerolinecolor='darkgreen'
    )
)

st.plotly_chart(fig_cat, use_container_width=True)

# Total Hours Worked Over Time
hours_time = filtered_data.groupby('year_month')['Hours'].sum().reset_index()
hours_time['year_month'] = hours_time['year_month'].astype(str)

fig_hours = go.Figure()
fig_hours.add_trace(go.Scatter(
    x=hours_time['year_month'],
    y=hours_time['Hours'],
    mode='lines+markers',
    line=dict(color='lightgreen', width=3),
    marker=dict(size=8, color='green'),
    hovertemplate='Month: %{x}<br>Hours: %{y:.2f}<extra></extra>'
))

fig_hours.update_layout(
    title='Total Hours Worked Over Time',
    xaxis_title='Year-Month',
    yaxis_title='Total Hours',
    plot_bgcolor='black',
    font=dict(family='Arial', size=14, color='lightgreen'),
    margin=dict(l=40, r=40, t=70, b=50),
    xaxis=dict(showgrid=True, gridcolor='darkgreen'),
    yaxis=dict(showgrid=True, gridcolor='darkgreen')
)

st.plotly_chart(fig_hours, use_container_width=True)

# Filtered Data Table (limited to max_rows)
st.subheader("📄 Filtered Task Data Table")
st.dataframe(
    filtered_data[['Full_Name', 'started_at', 'task', 'Lemmatized_Task', 'Hours', 'Categorized']].head(max_rows),
    use_container_width=True
)

# Show Lemmatization Highlights
with st.expander("🔍 Lemmatization Word Changes", expanded=False):
    for i in range(min(10, filtered_data.shape[0])):
        row = filtered_data.iloc[i]
        st.markdown(f"**Original Task:** {row['task']}")
        st.markdown(f"**Lemmatization Changes:** {' '.join(row['lemmatization_changes'])}")
        st.markdown("---")

# Top 50 Most Common Lemmatized Words
with st.expander("🔍 Top 50 Most Common Words (Lemmatized)", expanded=False):
    all_words = [word for sublist in filtered_data['task_wo_punct_split_wo_stopwords_lemmatized'] for word in sublist]
    word_counts = Counter(all_words).most_common(50)
    if word_counts:
        words, counts = zip(*word_counts)
        df_plot = pd.DataFrame({'Word': words, 'Count': counts})
        fig = go.Figure(go.Bar(
            x=df_plot['Word'],
            y=df_plot['Count'],
            marker=dict(
                color=df_plot['Count'],
                colorscale='Greens',
                line=dict(width=0.5, color='black')
            ),
            hovertemplate='Word: %{x}<br>Count: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title="Top 50 Most Common Words (Lemmatized)",
            xaxis_title="Word",
            yaxis_title="Frequency",
            xaxis_tickangle=-45,
            plot_bgcolor='black',
            font=dict(family='Arial', size=14, color='lightgreen'),
            margin=dict(l=40, r=40, t=70, b=120),
            yaxis=dict(gridcolor='darkgreen'),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No word frequency data available for the selected filters.")
