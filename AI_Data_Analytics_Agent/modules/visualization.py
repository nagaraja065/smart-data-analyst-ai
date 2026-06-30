import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np


def show_charts(df):
    st.subheader("📊 Data Visualizations")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    chart_type = st.selectbox(
        "Choose Chart Type",
        ["Bar Chart", "Line Chart", "Pie Chart", "Histogram",
         "Scatter Plot", "Box Plot", "Correlation Heatmap"]
    )

    if chart_type == "Bar Chart":
        _bar_chart(df, num_cols, cat_cols)

    elif chart_type == "Line Chart":
        _line_chart(df, num_cols)

    elif chart_type == "Pie Chart":
        _pie_chart(df, num_cols, cat_cols)

    elif chart_type == "Histogram":
        _histogram(df, num_cols)

    elif chart_type == "Scatter Plot":
        _scatter_plot(df, num_cols)

    elif chart_type == "Box Plot":
        _box_plot(df, num_cols, cat_cols)

    elif chart_type == "Correlation Heatmap":
        _heatmap(df, num_cols)


def _bar_chart(df, num_cols, cat_cols):
    if not cat_cols or not num_cols:
        st.warning("Need at least one categorical and one numerical column.")
        return
    col1, col2 = st.columns(2)
    x = col1.selectbox("X axis (Category)", cat_cols)
    y = col2.selectbox("Y axis (Numeric)", num_cols)
    agg = st.radio("Aggregation", ["Sum", "Mean", "Count"], horizontal=True)

    if agg == "Sum":
        data = df.groupby(x)[y].sum().reset_index()
    elif agg == "Mean":
        data = df.groupby(x)[y].mean().reset_index()
    else:
        data = df.groupby(x)[y].count().reset_index()

    data = data.sort_values(y, ascending=False)
    fig = px.bar(data, x=x, y=y, color=x, title=f"{agg} of {y} by {x}",
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def _line_chart(df, num_cols):
    date_col = None
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            date_col = col
            break

    if date_col:
        try:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
            y_col = st.selectbox("Y axis", num_cols)
            monthly = df_copy.groupby(df_copy[date_col].dt.to_period("M"))[y_col].sum().reset_index()
            monthly[date_col] = monthly[date_col].astype(str)
            fig = px.line(monthly, x=date_col, y=y_col,
                          title=f"Monthly {y_col} Trend", markers=True,
                          template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("No date column found. Showing index-based line chart.")
        y_col = st.selectbox("Y axis", num_cols)
        fig = px.line(df.reset_index(), x="index", y=y_col, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


def _pie_chart(df, num_cols, cat_cols):
    if not cat_cols or not num_cols:
        st.warning("Need categorical and numerical columns.")
        return
    col1, col2 = st.columns(2)
    names = col1.selectbox("Category", cat_cols)
    values = col2.selectbox("Values", num_cols)
    top_n = st.slider("Top N categories", 3, 15, 6)

    data = df.groupby(names)[values].sum().nlargest(top_n).reset_index()
    fig = px.pie(data, names=names, values=values,
                 title=f"{values} by {names} (Top {top_n})",
                 template="plotly_white", hole=0.3)
    st.plotly_chart(fig, use_container_width=True)


def _histogram(df, num_cols):
    col = st.selectbox("Select column", num_cols)
    bins = st.slider("Number of bins", 10, 100, 30)
    fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}",
                       template="plotly_white", color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig, use_container_width=True)


def _scatter_plot(df, num_cols):
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
        return
    col1, col2, col3 = st.columns(3)
    x = col1.selectbox("X axis", num_cols)
    y = col2.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1))
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    color = col3.selectbox("Color by", ["None"] + cat_cols)

    fig = px.scatter(df, x=x, y=y,
                     color=color if color != "None" else None,
                     title=f"{x} vs {y}", template="plotly_white",
                     opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)


def _box_plot(df, num_cols, cat_cols):
    col1, col2 = st.columns(2)
    y = col1.selectbox("Numeric column", num_cols)
    x = col2.selectbox("Group by", ["None"] + cat_cols)

    fig = px.box(df, y=y, x=x if x != "None" else None,
                 title=f"Box Plot: {y}", template="plotly_white",
                 color=x if x != "None" else None)
    st.plotly_chart(fig, use_container_width=True)


def _heatmap(df, num_cols):
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
        return
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
