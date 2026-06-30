import streamlit as st
import pandas as pd


def load_dataset():
    st.subheader("📁 Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, Excel (.xlsx, .xls)"
    )

    df = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ File uploaded: **{uploaded_file.name}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("📋 Rows", f"{df.shape[0]:,}")
            col2.metric("📊 Columns", df.shape[1])
            col3.metric("🗂️ File Size", f"{uploaded_file.size / 1024:.1f} KB")

            with st.expander("👀 Preview Dataset (First 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)

            with st.expander("🔍 Column Info"):
                info_df = pd.DataFrame({
                    "Column": df.columns,
                    "Data Type": df.dtypes.values,
                    "Non-Null Count": df.notnull().sum().values,
                    "Null Count": df.isnull().sum().values,
                    "Null %": (df.isnull().sum().values / len(df) * 100).round(2)
                })
                st.dataframe(info_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

    else:
        st.info("👆 Upload a CSV or Excel file to get started. A sample dataset is available below.")

        if st.button("📥 Load Sample Sales Dataset"):
            try:
                df = pd.read_csv("data/sample_sales.csv")
                st.success("✅ Sample dataset loaded!")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.warning(f"Sample file not found: {e}")

    return df
