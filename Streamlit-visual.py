import plotly.graph_objects as go

# --- DASHBOARD / OVERVIEW ---

st.title("📊 Task Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Tasks", filtered_data.shape[0])
col2.metric("Total Hours", round(filtered_data["Hours"].sum(), 2))
col3.metric("Unique Users", filtered_data["Full_Name"].nunique())
col4.metric("Unique Projects", filtered_data["ProjectID"].nunique())

# Tasks by Category (exploded)
cat_counts = filtered_data.explode('Categorized')['Categorized'].value_counts()

# Modern bar chart with custom colors and rounded bars
fig_cat = go.Figure()
fig_cat.add_trace(go.Bar(
    x=cat_counts.index,
    y=cat_counts.values,
    marker=dict(
        color=cat_counts.values,
        colorscale='Tealrose',
        line=dict(width=0.8, color='DarkSlateGrey'),
        # Rounded bars not directly supported in plotly, but we can soften appearance by layout
    ),
    hovertemplate='Category: %{x}<br>Tasks: %{y}<extra></extra>'
))

fig_cat.update_layout(
    title='Task Counts by Category',
    xaxis_title='Category',
    yaxis_title='Number of Tasks',
    xaxis_tickangle=-45,
    plot_bgcolor='white',
    font=dict(family='Arial', size=14, color='black'),
    margin=dict(l=40, r=40, t=70, b=100),
    yaxis=dict(
        gridcolor='LightGray',
        zeroline=True,
        zerolinecolor='LightGray'
    )
)

st.plotly_chart(fig_cat, use_container_width=True)

# Hours over Time (monthly)
hours_time = filtered_data.groupby('year_month')['Hours'].sum().reset_index()
hours_time['year_month'] = hours_time['year_month'].astype(str)

fig_time = go.Figure()

fig_time.add_trace(go.Scatter(
    x=hours_time['year_month'],
    y=hours_time['Hours'],
    mode='lines+markers',
    line=dict(color='teal', width=3, shape='spline', smoothing=1.3),
    marker=dict(size=8, color='darkcyan'),
    hovertemplate='Month: %{x}<br>Hours: %{y:.2f}<extra></extra>'
))

fig_time.update_layout(
    title='Total Hours Worked Over Time',
    xaxis_title='Year-Month',
    yaxis_title='Total Hours',
    plot_bgcolor='white',
    font=dict(family='Arial', size=14, color='black'),
    margin=dict(l=40, r=40, t=70, b=50),
    xaxis=dict(
        showgrid=True,
        gridcolor='LightGray',
        tickangle=-45
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='LightGray',
        zeroline=True,
        zerolinecolor='LightGray'
    )
)

st.plotly_chart(fig_time, use_container_width=True)

# Top 50 Most Common Lemmatized Words
with st.expander("🔍 Top 50 Most Common Searches (Lemmatized)", expanded=True):
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
                colorscale='Viridis',
                line=dict(width=0.5, color='black')
            ),
            hovertemplate='Word: %{x}<br>Count: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title="Top 50 Most Common Lemmatized Words",
            xaxis_title="Word",
            yaxis_title="Frequency",
            xaxis_tickangle=-45,
            plot_bgcolor='white',
            font=dict(family='Arial', size=14, color='black'),
            margin=dict(l=40, r=40, t=70, b=120),
            yaxis=dict(gridcolor='LightGray'),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = go.Figure()
        fig.update_layout(
            title="No Data Found for Search Frequency",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text="No search data available",
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )],
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
