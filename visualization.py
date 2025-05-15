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

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

# Streamlit page configuration
st.set_page_config(page_title="Task Dashboard", layout="wide")

# Define color palette
color_palette = sns.color_palette("autumn", as_cmap=True)

# Load data
@st.cache_data
def load_data():
    # Replace this with actual data loading logic
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]

    # Check if any CSV files are found
    if not csv_files:
        print("No CSV files found in the repository.")
        return pd.DataFrame()  # Return an empty DataFrame if no files are found

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

    # Additional preprocessing logic
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
            "miscellaneous": []  # For words that don't fit into other categories
        }
    
    def categorize_words(task_wo_punct_split_wo_stopwords_lemmatized, categories):
        matched_categories = set()

        for word in task_wo_punct_split_wo_stopwords_lemmatized:
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

# Load the data
combined_df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
categories = st.sidebar.multiselect("Select Categories", options=combined_df['Categorized'].explode().unique())
date_filter = st.sidebar.date_input("Filter by Date", [])
search_term = st.sidebar.text_input("Search Task", "")
full_name_filter = st.sidebar.multiselect("Filter by Full Name", options=combined_df['Full_Name'].unique())

# Filter data
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
if search_term:
    filtered_data = filtered_data[filtered_data['task'].str.contains(search_term, case=False, na=False)]
if full_name_filter:
    filtered_data = filtered_data[filtered_data['Full_Name'].isin(full_name_filter)]

# Convert filtered data to CSV
csv_data = filtered_data.to_csv(index=False).encode('utf-8')

# Sidebar download button
st.sidebar.download_button(
    label="📥 Download Filtered CSV",
    data=csv_data,
    file_name="filtered_data.csv",
    mime="text/csv"
)

# Tabs for graphs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "Hours", "Entries", "4", "5", "6", "7"])

# Tab 1: Overview
with tab1:
    st.subheader("Preview of Filtered Data (First 100 Rows)")
    st.dataframe(filtered_data.head(100), use_container_width=True)

    st.subheader("Missing Values by Column")
    missing_counts = filtered_data.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if not missing_counts.empty:
        missing_df = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing Values': missing_counts.values
        })

        fig_missing = px.bar(
            missing_df,
            x='Column',
            y='Missing Values',
            color='Missing Values',
            color_continuous_scale='Reds',
            title="Number of Missing Values per Column",
            labels={'Missing Values': 'Count of NaNs'},
            hover_data={'Column': True, 'Missing Values': True}
        )
        fig_missing.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("No missing values found in the filtered dataset.")

    all_words = [word for sublist in filtered_data['task_wo_punct_split_wo_stopwords_lemmatized'] for word in sublist]
    word_counts = Counter(all_words).most_common(20)

    if word_counts:
        words, counts = zip(*word_counts)
        df_plot = pd.DataFrame({'Word': words, 'Count': counts})
        fig = px.bar(
            df_plot,
            x='Word',
            y='Count',
            color='Count',
            color_continuous_scale='Oranges',
            title="Top 20 Most Common Words (Lemmatized)",
            labels={'Count': 'Frequency'},
            hover_data={'Word': True, 'Count': True}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Empty plot using Plotly to stay consistent
        fig = px.bar(title="Top 20 Most Common Words (Lemmatized)")
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': "No data available",
                'xref': "paper",
                'yref': "paper",
                'showarrow': False,
                'font': {'size': 16}
            }]
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Hours by Time Period
with tab2:
    st.header("Hours by Time Period")
    time_option = st.selectbox("Select Time Period", options=["Week", "Month", "Year"], index=2)

    if time_option == "Week":
        time_col = "week"
    elif time_option == "Month":
        time_col = "month"
    else:
        time_col = "year"

    grouped_data = filtered_data.groupby([time_col, 'Full_Name'])['Hours'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=grouped_data, x=time_col, y="Hours", hue="Full_Name", palette="autumn", ax=ax, dodge=False)
    ax.set_title(f"Hours by {time_option}", fontsize=14)
    ax.set_xlabel(time_option, fontsize=12)
    ax.set_ylabel("Hours", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    fig = px.bar(grouped_data, x=time_col, y="Hours", title="Hours Logged by User", color="Full_Name",
                 color_discrete_sequence=px.colors.qualitative.Plotly, barmode='group')

    fig.update_layout(
        plot_bgcolor='white',     # background of the plot area
    )
    st.plotly_chart(fig)

# Tab 3: Entries Count by Time Period - Alternative View
with tab3:
    st.header("Entries Count by Time Period - Alternative View")
    counts_time_option = st.selectbox("Select Time Period", options=["Week", "Month", "Year"], index=1, key="counts_time_option_alt")

    if counts_time_option == "Week":
        count_col = "week"
    elif counts_time_option == "Month":
        count_col = "year_month"
    else:
        count_col = "year"

    counts = filtered_data.groupby(count_col)['ProjectID-ID'].nunique().reset_index(name='Unique Entries')

    # For year_month, convert Period to string for Plotly compatibility
    if count_col == 'year_month':
        counts[count_col] = counts[count_col].astype(str)

    fig_line = px.line(
        counts,
        x=count_col,
        y='Unique Entries',
        markers=True,
        title=f"Unique Entries Trend by {counts_time_option}",
        labels={count_col: counts_time_option, 'Unique Entries': 'Unique Entries Count'},
        line_shape='spline'  # smooth curves
    )
    fig_line.update_traces(line=dict(color='#FE6E00'), marker=dict(size=8))
    fig_line.update_layout(
        plot_bgcolor='white',
        xaxis=dict(tickangle=-45),
        yaxis=dict(gridcolor='lightgrey'),
        hovermode='x unified'
    )
    st.plotly_chart(fig_line, use_container_width=True)
