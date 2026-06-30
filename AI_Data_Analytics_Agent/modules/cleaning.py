import streamlit as st
import pandas as pd


def validate_data(df):
    st.subheader("🔍 Data Validation Report")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    issues = []
    if df.isnull().sum().sum() > 0:
        issues.append(f"⚠️ {int(df.isnull().sum().sum())} missing values detected")
    if df.duplicated().sum() > 0:
        issues.append(f"⚠️ {int(df.duplicated().sum())} duplicate rows found")

    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success("✅ No data quality issues found!")

    with st.expander("📊 Missing Values by Column"):
        null_df = df.isnull().sum().reset_index()
        null_df.columns = ["Column", "Missing Count"]
        null_df["Missing %"] = (null_df["Missing Count"] / len(df) * 100).round(2)
        null_df = null_df[null_df["Missing Count"] > 0]
        if not null_df.empty:
            st.dataframe(null_df, use_container_width=True)
        else:
            st.write("No missing values.")

    return df


def clean_data(df):
    st.subheader("🧹 Data Cleaning")

    original_shape = df.shape
    df_clean = df.copy()

    actions = []

    # Remove duplicates
    dupes = df_clean.duplicated().sum()
    if dupes > 0:
        df_clean = df_clean.drop_duplicates()
        actions.append(f"✅ Removed {dupes} duplicate rows")

    # Fill missing numeric values with median
    num_cols = df_clean.select_dtypes(include="number").columns
    for col in num_cols:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            actions.append(f"✅ Filled {missing} missing values in **{col}** with median")

    # Fill missing categorical values with mode
    cat_cols = df_clean.select_dtypes(include="object").columns
    for col in cat_cols:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
            df_clean[col] = df_clean[col].fillna(mode_val)
            actions.append(f"✅ Filled {missing} missing values in **{col}** with '{mode_val}'")

    # Try to parse date columns
    for col in df_clean.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df_clean[col] = pd.to_datetime(df_clean[col])
                actions.append(f"✅ Converted **{col}** to datetime")
            except Exception:
                pass

    if actions:
        for action in actions:
            st.markdown(action)
    else:
        st.info("No cleaning actions needed — data is already clean!")

    col1, col2 = st.columns(2)
    col1.metric("Before Cleaning", f"{original_shape[0]} rows")
    col2.metric("After Cleaning", f"{df_clean.shape[0]} rows")

    return df_clean
