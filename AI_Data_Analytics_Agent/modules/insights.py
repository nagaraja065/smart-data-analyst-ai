import streamlit as st
import pandas as pd
import json
import requests


def generate_insights(df):
    st.subheader("🤖 AI Business Insights")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Build a stats summary to send to Claude
    summary = _build_summary(df, num_cols, cat_cols)

    st.info("Click below to generate AI-powered business insights from your dataset.")

    if st.button("✨ Generate AI Insights"):
        api_key = st.session_state.get("api_key", "")
        with st.spinner("Analyzing your data with AI..."):
            if api_key:
                insights = _call_claude(summary, df.columns.tolist(), len(df), api_key)
            else:
                insights = _generate_local_insights(df, num_cols, cat_cols)
            if insights:
                st.markdown("### 📋 Business Insights")
                st.markdown(insights)


def _build_summary(df, num_cols, cat_cols):
    summary_parts = []

    summary_parts.append(f"Dataset has {len(df)} rows and {df.shape[1]} columns.")

    # Numeric stats
    if num_cols:
        stats = df[num_cols].describe().T[["mean", "min", "max", "std"]].round(2)
        summary_parts.append("Numeric column stats:\n" + stats.to_string())

    # Categorical top values
    for col in cat_cols[:3]:
        top = df[col].value_counts().head(3)
        summary_parts.append(f"Top values in '{col}': {top.to_dict()}")

    # Sales/profit summary if columns exist
    for keyword in ["sales", "profit", "revenue", "quantity"]:
        for col in num_cols:
            if keyword in col.lower():
                summary_parts.append(f"Total {col}: {df[col].sum():,.2f}, Average: {df[col].mean():,.2f}")

    return "\n".join(summary_parts)


def _call_claude(summary, columns, row_count, api_key):
    try:
        prompt = f"""You are a senior data analyst. Based on the following dataset summary, generate 6-8 clear and actionable business insights. 
Format each insight as a bullet point. Be specific with numbers where possible. 
End with 2-3 strategic recommendations.

Dataset columns: {', '.join(columns)}
Total records: {row_count}

Data Summary:
{summary}

Generate insights in this format:
**Key Findings:**
- [insight 1]
- [insight 2]
...

**Strategic Recommendations:**
- [recommendation 1]
- [recommendation 2]
"""
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        data = response.json()
        if "content" in data and data["content"]:
            return data["content"][0].get("text", "No insights generated.")
        else:
            error_msg = data.get("error", {}).get("message", "Unknown error")
            st.warning(f"API Error: {error_msg}. Falling back to local insights.")
            return None
    except Exception as e:
        st.warning(f"Could not reach Claude API: {e}. Falling back to local insights.")
        return None


def _generate_local_insights(df, num_cols, cat_cols):
    """Generate insights without API using data analysis."""
    insights = ["**Key Findings:**\n"]

    # Overall stats
    insights.append(f"- 📊 Dataset contains **{len(df):,}** records across **{df.shape[1]}** columns")

    # Numeric insights
    for col in num_cols:
        if any(kw in col.lower() for kw in ["sales", "revenue", "amount", "total"]):
            total = df[col].sum()
            avg = df[col].mean()
            insights.append(f"- 💰 Total **{col}**: {total:,.2f} with an average of {avg:,.2f} per record")
        if any(kw in col.lower() for kw in ["profit", "margin"]):
            total = df[col].sum()
            neg = (df[col] < 0).sum()
            insights.append(f"- 📈 Total **{col}**: {total:,.2f}")
            if neg > 0:
                insights.append(f"- ⚠️ **{neg}** transactions have negative {col} — potential loss areas")

    # Top performers
    if cat_cols and num_cols:
        for cat in cat_cols[:2]:
            for num in num_cols[:2]:
                top = df.groupby(cat)[num].sum().idxmax()
                top_val = df.groupby(cat)[num].sum().max()
                insights.append(f"- 🏆 Top **{cat}** by {num}: **{top}** ({top_val:,.2f})")

    # Recommendations
    insights.append("\n**Strategic Recommendations:**")
    insights.append("- Focus resources on top-performing segments to maximize ROI")
    insights.append("- Investigate underperforming categories for improvement opportunities")
    insights.append("- Implement regular data-driven review cycles for continuous optimization")

    return "\n".join(insights)
