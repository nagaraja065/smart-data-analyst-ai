import streamlit as st
import pandas as pd
from modules.ai_agent import DataAgent


def show_agent_tab(df):
    """Render the AI Agent tab with chat interface and quick actions."""

    st.markdown("""
    <style>
        .agent-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1rem;
        }
        .agent-header h2 {
            color: white !important;
            margin: 0;
        }
        .agent-header p {
            color: rgba(255,255,255,0.85);
            margin: 0.3rem 0 0 0;
        }
        .step-badge {
            display: inline-block;
            background: #f0f2f6;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            margin: 2px 4px;
            color: #333;
        }
        .agent-status {
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 8px 12px;
            border-radius: 0 8px 8px 0;
            margin: 8px 0;
            font-size: 0.85rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div class="agent-header">
        <h2>🤖 AI Data Agent</h2>
        <p>Ask anything about your data — I'll analyze, compute, and visualize for you.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Initialize Agent ──
    api_key = st.session_state.get("api_key", "")
    agent = DataAgent(df, api_key=api_key if api_key else None)

    # ── Agent Status ──
    if api_key:
        st.markdown('<div class="agent-status">🟢 <strong>Claude AI Connected</strong> — Enhanced analysis enabled</div>', unsafe_allow_html=True)
    else:
        st.info("💡 Running in **local mode** (no API key needed). Add a Claude API key in the sidebar for deeper AI-powered analysis.")

    # ── Quick Action Buttons ──
    st.markdown("#### ⚡ Quick Actions")
    qa_cols = st.columns(4)

    quick_actions = [
        ("📊 Full Analysis", "quick_analyze"),
        ("🚨 Find Anomalies", "quick_anomalies"),
        ("🎨 Suggest Charts", "quick_suggest_charts"),
        ("💡 Business Insights", "quick_insights"),
    ]

    for i, (label, action) in enumerate(quick_actions):
        if qa_cols[i].button(label, key=f"qa_{action}", use_container_width=True):
            with st.spinner("🤖 Agent is thinking..."):
                method = getattr(agent, action)
                result = method()
                _add_to_history("user", f"[Quick Action] {label}")
                _add_to_history("assistant", result["text"], result.get("chart"), result.get("steps"))
            st.rerun()

    st.divider()

    # ── Chat History ──
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []

    for msg in st.session_state.agent_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart"):
                st.plotly_chart(msg["chart"], use_container_width=True)
            if msg.get("steps"):
                with st.expander("🔍 Agent Reasoning Steps"):
                    for step in msg["steps"]:
                        st.markdown(f"<span class='step-badge'>{step}</span>", unsafe_allow_html=True)

    # ── Suggestion Chips (shown when no history) ──
    if not st.session_state.agent_history:
        st.markdown("#### 💬 Try asking...")
        chip_cols = st.columns(2)
        suggestions = [
            "Give me an overview of this dataset",
            "What are the top 5 products by sales?",
            "Show me the sales trend over time",
            "Find outliers and anomalies in the data",
        ]
        for i, suggestion in enumerate(suggestions):
            if chip_cols[i % 2].button(f"💬 {suggestion}", key=f"sug_{i}", use_container_width=True):
                with st.spinner("🤖 Agent is thinking..."):
                    result = agent.query(suggestion)
                    _add_to_history("user", suggestion)
                    _add_to_history("assistant", result["text"], result.get("chart"), result.get("steps"))
                st.rerun()

    # ── Chat Input ──
    user_input = st.chat_input("Ask the AI Agent anything about your data...")
    if user_input:
        _add_to_history("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Agent is analyzing your data..."):
                result = agent.query(user_input)

            st.markdown(result["text"])
            if result.get("chart"):
                st.plotly_chart(result["chart"], use_container_width=True)
            if result.get("steps"):
                with st.expander("🔍 Agent Reasoning Steps"):
                    for step in result["steps"]:
                        st.markdown(f"<span class='step-badge'>{step}</span>", unsafe_allow_html=True)

            _add_to_history("assistant", result["text"], result.get("chart"), result.get("steps"))

    # ── Clear Chat ──
    if st.session_state.agent_history:
        st.divider()
        if st.button("🗑️ Clear Agent Chat", key="clear_agent"):
            st.session_state.agent_history = []
            st.rerun()


def _add_to_history(role, content, chart=None, steps=None):
    """Add a message to the agent chat history."""
    st.session_state.agent_history.append({
        "role": role,
        "content": content,
        "chart": chart,
        "steps": steps,
    })
