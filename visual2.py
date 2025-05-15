import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

st.title("Task Log Visualization")

# Sidebar filters
users = st.sidebar.multiselect("Select Users", options=df["username"].unique(), default=df["username"].unique())
df = df[df["username"].isin(users)]

# Tabs for navigation
tabs = st.tabs(["Overview", "Tasks by Category", "Daily Trend", "Common Search", "Common Task", "Common Words", "Top 50 Searches"])

# Overview Tab
with tabs[0]:
    st.header("Overview")
    st.metric("Total Tasks Logged", len(df))
    st.metric("Total Users", df["username"].nunique())

# Tasks by Category Tab
with tabs[1]:
    st.header("Tasks by Category")
    category_counts = df["category"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=category_counts.values, y=category_counts.index, ax=ax)
    ax.set_xlabel("Number of Tasks")
    ax.set_ylabel("Category")
    st.pyplot(fig)

# Daily Trend Tab
with tabs[2]:
    st.header("Daily Tasks Trend")
    daily_counts = df.groupby(df["date"].dt.date).size()
    fig, ax = plt.subplots()
    daily_counts.plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Tasks")
    st.pyplot(fig)

# Common Search Tab
with tabs[3]:
    st.header("Most Common Search Strings")
    search_counts = df["task"].value_counts().head(20)
    st.bar_chart(search_counts)

# Common Task Tab
with tabs[4]:
    st.header("Most Common Raw Words in Tasks")
    all_words = " ".join(df["task"].astype(str)).lower()
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z ]", "", all_words))
    filtered_tokens = [word for word in tokens if word not in stopwords.words("english") and len(word) > 2]
    word_freq = Counter(filtered_tokens)
    common_words = pd.DataFrame(word_freq.most_common(20), columns=["Word", "Frequency"])
    fig, ax = plt.subplots()
    sns.barplot(data=common_words, x="Frequency", y="Word", ax=ax)
    st.pyplot(fig)

# Common Words (Lemmatized) Tab
with tabs[5]:
    st.header("Most Common Lemmatized Words")
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    lemmatized_freq = Counter(lemmatized_words)
    lemmatized_df = pd.DataFrame(lemmatized_freq.most_common(20), columns=["Word", "Frequency"])
    fig, ax = plt.subplots()
    sns.barplot(data=lemmatized_df, x="Frequency", y="Word", ax=ax)
    st.pyplot(fig)

# Top 50 Searches Tab (Moved to bottom)
with tabs[6]:
    st.header("Top 50 Most Frequent Search Strings")
    top_50_searches = df["task"].value_counts().head(50)
    st.bar_chart(top_50_searches)
