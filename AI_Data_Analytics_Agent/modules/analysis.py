import streamlit as st
import pandas as pd
import numpy as np


def run_eda(df):
    st.subheader("📊 Exploratory Data Analysis")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Statistical Summary
    with st.expander("📈 Statistical Summary", expanded=True):
        if num_cols:
            st.dataframe(df[num_cols].describe().T.round(2), use_container_width=True)

    # Categorical summaries
    if cat_cols:
        with st.expander("🗂️ Categorical Column Summaries"):
            selected_cat = st.selectbox("Select a categorical column", cat_cols)
            val_counts = df[selected_cat].value_counts().reset_index()
            val_counts.columns = [selected_cat, "Count"]
            val_counts["Percentage"] = (val_counts["Count"] / len(df) * 100).round(2)
            st.dataframe(val_counts, use_container_width=True)

    # Business KPIs
    st.subheader("💼 Business KPIs")
    _show_kpis(df, num_cols, cat_cols)


def _show_kpis(df, num_cols, cat_cols):
    # Try to detect sales/profit/quantity columns
    sales_col = _find_col(df, ["sales", "revenue", "amount", "total"])
    profit_col = _find_col(df, ["profit", "margin", "income"])
    qty_col = _find_col(df, ["quantity", "qty", "units"])
    date_col = _find_col(df, ["date", "time", "order_date"])
    cat_col = _find_col(df, ["category", "type", "segment"])
    region_col = _find_col(df, ["region", "state", "location", "city", "area"])
    product_col = _find_col(df, ["product", "item", "name", "sku"])

    cols = st.columns(4)

    if sales_col:
        cols[0].metric("💰 Total Sales", f"₹{df[sales_col].sum():,.0f}")
        cols[1].metric("📊 Avg Sale", f"₹{df[sales_col].mean():,.0f}")

    if profit_col:
        cols[2].metric("📈 Total Profit", f"₹{df[profit_col].sum():,.0f}")
        if sales_col and df[sales_col].sum() > 0:
            margin = df[profit_col].sum() / df[sales_col].sum() * 100
            cols[3].metric("🎯 Profit Margin", f"{margin:.1f}%")

    # Top performers
    st.markdown("---")
    perf_cols = st.columns(3)

    if product_col and sales_col:
        top_product = df.groupby(product_col)[sales_col].sum().idxmax()
        perf_cols[0].metric("🏆 Top Product", top_product)

    if region_col and sales_col:
        top_region = df.groupby(region_col)[sales_col].sum().idxmax()
        perf_cols[1].metric("📍 Best Region", top_region)

    if cat_col and sales_col:
        top_cat = df.groupby(cat_col)[sales_col].sum().idxmax()
        perf_cols[2].metric("🗂️ Top Category", top_cat)

    # Monthly trend if date available
    if date_col and sales_col:
        st.markdown("---")
        st.markdown("**📅 Monthly Sales Trend**")
        try:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
            monthly = df_copy.groupby(df_copy[date_col].dt.to_period("M"))[sales_col].sum()
            monthly.index = monthly.index.astype(str)
            st.bar_chart(monthly)
        except Exception:
            pass


def _find_col(df, keywords):
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in col.lower():
                return col
    return None
