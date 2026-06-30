import streamlit as st
import pandas as pd
import requests
import json


def chat_with_data(df):
    st.subheader("💬 Chat with Your Data")
    st.caption("Ask questions about your dataset in plain English.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggestion buttons
    if not st.session_state.chat_history:
        st.markdown("**Quick questions to get started:**")
        cols = st.columns(2)
        suggestions = [
            "Which product sold the most?",
            "What is the total revenue?",
            "Which region has the highest profit?",
            "Show me the top 5 customers by sales."
        ]
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"suggest_{i}"):
                _ask_question(df, s)
                st.rerun()

    # Chat input
    user_input = st.chat_input("Ask anything about your data...")
    if user_input:
        _ask_question(df, user_input)
        st.rerun()

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


def _ask_question(df, question):
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Build dataset context
    context = _build_context(df)

    api_key = st.session_state.get("api_key", "")

    if api_key:
        answer = _ask_claude(context, question, api_key)
    else:
        answer = _ask_local(df, question, context)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})


def _ask_claude(context, question, api_key):
    """Ask Claude with proper API headers."""
    prompt = f"""You are a data analyst assistant. Answer the user's question about the dataset.
Be concise, use numbers/percentages where relevant, and format nicely with bullet points if listing multiple items.
If the question can't be answered from the data, say so politely.

Dataset Context:
{context}

User Question: {question}

Answer:"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        data = response.json()
        if "content" in data and data["content"]:
            return data["content"][0].get("text", "I couldn't generate an answer.")
        else:
            error_msg = data.get("error", {}).get("message", "Unknown error")
            return f"API Error: {error_msg}. Try removing and re-entering your API key."
    except Exception as e:
        return f"Error: {e}"


def _ask_local(df, question, context):
    """Answer questions locally without API."""
    q = question.lower()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Find relevant column
    target_col = None
    for col in df.columns:
        if col.lower() in q or any(word in q for word in col.lower().replace("_", " ").split() if len(word) > 2):
            target_col = col
            break

    # Total / sum questions
    if any(kw in q for kw in ["total", "sum", "overall"]):
        for col in num_cols:
            if col.lower() in q or any(kw in col.lower() for kw in ["sales", "revenue", "profit", "amount"]):
                return f"📊 The total **{col}** is **{df[col].sum():,.2f}**\n\n- Average: {df[col].mean():,.2f}\n- Records: {len(df):,}"

    # Top / most / highest questions
    if any(kw in q for kw in ["top", "most", "highest", "best", "which"]):
        num_col = None
        for col in num_cols:
            if any(kw in col.lower() for kw in ["sales", "revenue", "profit", "quantity", "amount"]):
                num_col = col
                break
        if not num_col and num_cols:
            num_col = num_cols[0]

        if num_col:
            for cat in cat_cols:
                if cat.lower() in q or any(word in q for word in cat.lower().replace("_", " ").split() if len(word) > 2):
                    top5 = df.groupby(cat)[num_col].sum().nlargest(5).reset_index()
                    lines = [f"🏆 **Top 5 {cat} by {num_col}:**\n"]
                    for _, row in top5.iterrows():
                        lines.append(f"- **{row[cat]}**: {row[num_col]:,.2f}")
                    return "\n".join(lines)

            # Default: use first cat col
            if cat_cols:
                cat = cat_cols[0]
                top5 = df.groupby(cat)[num_col].sum().nlargest(5).reset_index()
                lines = [f"🏆 **Top 5 {cat} by {num_col}:**\n"]
                for _, row in top5.iterrows():
                    lines.append(f"- **{row[cat]}**: {row[num_col]:,.2f}")
                return "\n".join(lines)

    # Count questions
    if any(kw in q for kw in ["how many", "count", "number of"]):
        return f"📋 The dataset has **{len(df):,}** records and **{df.shape[1]}** columns."

    # Column-specific questions
    if target_col:
        col = df[target_col]
        if pd.api.types.is_numeric_dtype(col):
            return (f"📊 **{target_col}** Stats:\n\n"
                    f"- Total: {col.sum():,.2f}\n"
                    f"- Average: {col.mean():,.2f}\n"
                    f"- Min: {col.min():,.2f}\n"
                    f"- Max: {col.max():,.2f}")
        else:
            top = col.value_counts().head(5)
            lines = [f"📊 **{target_col}** Top Values:\n"]
            for val, count in top.items():
                lines.append(f"- **{val}**: {count} ({count/len(df)*100:.1f}%)")
            return "\n".join(lines)

    # Fallback
    return (f"📋 I analyzed your dataset with **{len(df):,}** rows and **{df.shape[1]}** columns.\n\n"
            f"Columns: {', '.join(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}\n\n"
            f"💡 *Try asking about specific columns like sales, profit, or categories. "
            f"For deeper analysis, add a Claude API key in the sidebar.*")


def _build_context(df):
    lines = [f"Columns: {', '.join(df.columns.tolist())}",
             f"Total rows: {len(df)}"]

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if num_cols:
        for col in num_cols:
            lines.append(f"{col}: total={df[col].sum():.2f}, mean={df[col].mean():.2f}, max={df[col].max():.2f}")

    for col in cat_cols[:3]:
        top = df[col].value_counts().head(5).to_dict()
        lines.append(f"{col} top values: {top}")

    return "\n".join(lines)
