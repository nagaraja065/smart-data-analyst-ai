import streamlit as st
import pandas as pd
import io
from datetime import datetime


def generate_report(df):
    st.subheader("📄 Download Reports")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Excel Report**")
        st.write("Full dataset + summary statistics")
        excel_data = _create_excel(df)
        st.download_button(
            label="⬇️ Download Excel Report",
            data=excel_data,
            file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col2:
        st.markdown("**📋 CSV Export**")
        st.write("Clean dataset as CSV")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name=f"clean_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    # Show summary table
    with st.expander("📊 Summary Statistics Preview"):
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            st.dataframe(df[num_cols].describe().T.round(2), use_container_width=True)


def _create_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Raw data
        df.to_excel(writer, sheet_name="Dataset", index=False)

        # Sheet 2: Summary stats
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            summary = df[num_cols].describe().T.round(2)
            summary.to_excel(writer, sheet_name="Summary Statistics")

        # Sheet 3: Categorical counts
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            row = 0
            ws = writer.book.create_sheet("Category Counts")
            for col in cat_cols:
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, "Count"]
                ws.cell(row=row+1, column=1, value=f"Column: {col}")
                row += 1
                for _, r in vc.iterrows():
                    ws.cell(row=row+1, column=1, value=r[col])
                    ws.cell(row=row+1, column=2, value=r["Count"])
                    row += 1
                row += 1

    return output.getvalue()
